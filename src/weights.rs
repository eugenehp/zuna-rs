/// Load pretrained ZUNA weights from a safetensors file (burn 0.20.1)
///
/// The HuggingFace file stores weights as bfloat16 under keys like
///   `model.encoder.tok_embeddings.weight`
///   `model.decoder.layers.0.cross_attention.wq.weight`
///
/// We strip the leading `model.` prefix and convert bf16 → f32.
///
/// ## Weight key reference
///
/// ENCODER
///   encoder.tok_embeddings.weight            [1024, 32]  no bias
///   encoder.registers                        [1, 32]     (Param, not Linear)
///   encoder.norm.inner.gamma                 [1024]      (RmsNorm)
///   encoder.output.weight                    [32, 1024]  no bias
///   encoder.layers.{i}.attention_norm.inner.gamma        [1024]
///   encoder.layers.{i}.attention.wq.weight               [1024,1024]
///   encoder.layers.{i}.attention.wk.weight               [1024,1024]
///   encoder.layers.{i}.attention.wv.weight               [1024,1024]
///   encoder.layers.{i}.attention.wo.weight               [1024,1024]
///   encoder.layers.{i}.ffn_norm.inner.gamma              [1024]
///   encoder.layers.{i}.feed_forward.w1.weight            [2816,1024]
///   encoder.layers.{i}.feed_forward.w2.weight            [1024,2816]
///   encoder.layers.{i}.feed_forward.w3.weight            [2816,1024]
///
/// DECODER
///   decoder.tok_embeddings.weight            [1024, 32]  no bias
///   decoder.t_embedder.weight                [32, 1]     (buffer)
///   decoder.t_embedder.proj.weight           [64, 64]    + bias [64]
///   decoder.encoder_proj.weight              [1024, 32]  + bias [1024]
///   decoder.norm.weight.weight               [1024, 64]  + bias [1024]   (AdaRMSNorm)
///   decoder.output.weight                    [32, 1024]  no bias
///   decoder.layers.{i}.cross_attention_x_norm.weight.weight  [1024,64]  + bias
///   decoder.layers.{i}.cross_attention_y_norm.weight.weight  [1024,64]  + bias
///   decoder.layers.{i}.cross_attention.wq/wk/wv/wo.weight    [1024,1024]
///   decoder.layers.{i}.attention_norm.weight.weight           [1024,64]  + bias
///   decoder.layers.{i}.attention.wq/wk/wv/wo.weight          [1024,1024]
///   decoder.layers.{i}.ffn_norm.weight.weight                 [1024,64]  + bias
///   decoder.layers.{i}.feed_forward.w1/w2/w3.weight          various

use std::collections::HashMap;
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;

use crate::model::encoder::EncoderTransformer;
use crate::model::decoder::DecoderTransformer;
use crate::model::encoder_decoder::EncoderDecoder;
use crate::config::ModelConfig;

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightMap {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st    = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::new();

        for (raw_key, view) in st.tensors() {
            let key = raw_key
                .strip_prefix("model.")
                .unwrap_or(raw_key.as_str())
                .to_string();

            let shape: Vec<usize> = view.shape().to_vec();
            let data  = view.data();

            let f32s: Vec<f32> = match view.dtype() {
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                other => anyhow::bail!("unsupported dtype {:?} for key {key}", other),
            };

            tensors.insert(key, (f32s, shape));
        }

        Ok(Self { tensors })
    }

    /// Load a tensor by its (prefix-stripped) safetensors key.
    pub fn get<B: Backend, const N: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self.tensors.get(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data.clone(), shape.clone()),
            device,
        ))
    }

    /// Infer the number of attention heads from the wq weight shape.
    /// wq is stored as [n_heads * head_dim, dim] (PyTorch out×in convention).
    pub fn infer_n_heads(&self, head_dim: usize) -> anyhow::Result<usize> {
        let key = "encoder.layers.0.attention.wq.weight";
        let (_, shape) = self.tensors.get(key)
            .ok_or_else(|| anyhow::anyhow!("key not found for n_heads inference: {key}"))?;
        // shape[0] = n_heads * head_dim  (PyTorch [out, in])
        Ok(shape[0] / head_dim)
    }

    pub fn print_keys(&self) {
        let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        keys.sort();
        for k in keys {
            let (_, s) = &self.tensors[k];
            println!("  {k:80}  {s:?}");
        }
    }
}

// ── Weight assignment helpers ─────────────────────────────────────────────────

/// Assign a 2-D weight tensor into a `burn::nn::Linear` (weight only).
///
/// PyTorch stores linear weights as [out, in]; burn's Row-layout stores them as
/// [in, out] and does `input @ weight`. So we must transpose from safetensors.
fn set_linear_w<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,   // raw safetensors tensor [out, in]
) {
    // Transpose [out, in] → [in, out] for burn's Row layout.
    linear.weight = linear.weight.clone().map(|_| w.transpose());
}

/// Assign weight + bias into a `burn::nn::Linear`.
fn set_linear_wb<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>,   // raw safetensors tensor [out, in]
    b: Tensor<B, 1>,
) {
    linear.weight = linear.weight.clone().map(|_| w.transpose());
    if let Some(ref bias) = linear.bias {
        linear.bias = Some(bias.clone().map(|_| b));
    }
}

/// Assign a 1-D weight into a `burn::nn::RmsNorm` (field: `gamma`).
fn set_rmsnorm<B: Backend>(norm: &mut burn::nn::RmsNorm<B>, w: Tensor<B, 1>) {
    norm.gamma = norm.gamma.clone().map(|_| w);
}

// ── Internal per-component loaders ───────────────────────────────────────────

/// Populate an already-constructed [`EncoderTransformer`] from a [`WeightMap`].
fn load_encoder_from_wm<B: Backend>(
    wm:     &WeightMap,
    enc:    &mut EncoderTransformer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    set_linear_wb(
        &mut enc.tok_embeddings,
        wm.get("encoder.tok_embeddings.weight", device)?,
        wm.get("encoder.tok_embeddings.bias",   device)?,
    );

    let regs: Tensor<B, 2> = wm.get("encoder.registers", device)?;
    enc.registers = enc.registers.clone().map(|_| regs);

    let norm_w: Tensor<B, 1> = wm.get("encoder.norm.weight", device)?;
    set_rmsnorm(&mut enc.norm.inner, norm_w);

    set_linear_w(&mut enc.output, wm.get("encoder.output.weight", device)?);

    for (i, layer) in enc.layers.iter_mut().enumerate() {
        let p = format!("encoder.layers.{i}");

        let an_w: Tensor<B, 1> = wm.get(&format!("{p}.attention_norm.weight"), device)?;
        set_rmsnorm(&mut layer.attention_norm.inner, an_w);

        set_linear_w(&mut layer.attention.wq, wm.get(&format!("{p}.attention.wq.weight"), device)?);
        set_linear_w(&mut layer.attention.wk, wm.get(&format!("{p}.attention.wk.weight"), device)?);
        set_linear_w(&mut layer.attention.wv, wm.get(&format!("{p}.attention.wv.weight"), device)?);
        set_linear_w(&mut layer.attention.wo, wm.get(&format!("{p}.attention.wo.weight"), device)?);

        let fn_w: Tensor<B, 1> = wm.get(&format!("{p}.ffn_norm.weight"), device)?;
        set_rmsnorm(&mut layer.ffn_norm.inner, fn_w);

        set_linear_w(&mut layer.feed_forward.w1, wm.get(&format!("{p}.feed_forward.w1.weight"), device)?);
        set_linear_w(&mut layer.feed_forward.w2, wm.get(&format!("{p}.feed_forward.w2.weight"), device)?);
        set_linear_w(&mut layer.feed_forward.w3, wm.get(&format!("{p}.feed_forward.w3.weight"), device)?);
    }
    Ok(())
}

/// Populate an already-constructed [`DecoderTransformer`] from a [`WeightMap`].
fn load_decoder_from_wm<B: Backend>(
    wm:     &WeightMap,
    dec:    &mut DecoderTransformer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    set_linear_wb(
        &mut dec.tok_embeddings,
        wm.get("decoder.tok_embeddings.weight", device)?,
        wm.get("decoder.tok_embeddings.bias",   device)?,
    );

    let fc_w: Tensor<B, 2> = wm.get("decoder.t_embedder.weight", device)?;
    dec.t_embedder.weight = dec.t_embedder.weight.clone().map(|_| fc_w);
    set_linear_wb(
        &mut dec.t_embedder.proj,
        wm.get("decoder.t_embedder.proj.weight", device)?,
        wm.get("decoder.t_embedder.proj.bias",   device)?,
    );

    set_linear_wb(
        &mut dec.encoder_proj,
        wm.get("decoder.encoder_proj.weight", device)?,
        wm.get("decoder.encoder_proj.bias",   device)?,
    );

    // AdaRMSNorm final norm — inner Linear is named "weight" in PyTorch.
    // safetensors keys: decoder.norm.weight.weight / decoder.norm.weight.bias
    set_linear_wb(
        &mut dec.norm.weight,
        wm.get("decoder.norm.weight.weight", device)?,
        wm.get("decoder.norm.weight.bias",   device)?,
    );

    set_linear_w(&mut dec.output, wm.get("decoder.output.weight", device)?);

    for (i, layer) in dec.layers.iter_mut().enumerate() {
        let p = format!("decoder.layers.{i}");

        set_linear_wb(&mut layer.cross_attention_x_norm.weight,
            wm.get(&format!("{p}.cross_attention_x_norm.weight.weight"), device)?,
            wm.get(&format!("{p}.cross_attention_x_norm.weight.bias"),   device)?);
        set_linear_wb(&mut layer.cross_attention_y_norm.weight,
            wm.get(&format!("{p}.cross_attention_y_norm.weight.weight"), device)?,
            wm.get(&format!("{p}.cross_attention_y_norm.weight.bias"),   device)?);

        set_linear_w(&mut layer.cross_attention.wq, wm.get(&format!("{p}.cross_attention.wq.weight"), device)?);
        set_linear_w(&mut layer.cross_attention.wk, wm.get(&format!("{p}.cross_attention.wk.weight"), device)?);
        set_linear_w(&mut layer.cross_attention.wv, wm.get(&format!("{p}.cross_attention.wv.weight"), device)?);
        set_linear_w(&mut layer.cross_attention.wo, wm.get(&format!("{p}.cross_attention.wo.weight"), device)?);

        set_linear_wb(&mut layer.attention_norm.weight,
            wm.get(&format!("{p}.attention_norm.weight.weight"), device)?,
            wm.get(&format!("{p}.attention_norm.weight.bias"),   device)?);
        set_linear_w(&mut layer.attention.wq, wm.get(&format!("{p}.attention.wq.weight"), device)?);
        set_linear_w(&mut layer.attention.wk, wm.get(&format!("{p}.attention.wk.weight"), device)?);
        set_linear_w(&mut layer.attention.wv, wm.get(&format!("{p}.attention.wv.weight"), device)?);
        set_linear_w(&mut layer.attention.wo, wm.get(&format!("{p}.attention.wo.weight"), device)?);

        set_linear_wb(&mut layer.ffn_norm.weight,
            wm.get(&format!("{p}.ffn_norm.weight.weight"), device)?,
            wm.get(&format!("{p}.ffn_norm.weight.bias"),   device)?);
        set_linear_w(&mut layer.feed_forward.w1, wm.get(&format!("{p}.feed_forward.w1.weight"), device)?);
        set_linear_w(&mut layer.feed_forward.w2, wm.get(&format!("{p}.feed_forward.w2.weight"), device)?);
        set_linear_w(&mut layer.feed_forward.w3, wm.get(&format!("{p}.feed_forward.w3.weight"), device)?);
    }
    Ok(())
}

// ── Public partial loaders ────────────────────────────────────────────────────

/// Load only the **encoder** weights from a safetensors file.
///
/// Skips all decoder tensors — faster startup and ~half the memory when only
/// embeddings are needed.
///
/// Returns `(encoder, n_heads)`.  `n_heads` is inferred from the weight shape
/// because `config.json` does not store it explicitly.
pub fn load_encoder_weights<B: Backend>(
    cfg:          &ModelConfig,
    weights_path: &str,
    device:       &B::Device,
) -> anyhow::Result<(EncoderTransformer<B>, usize)> {
    let hidden_dim = cfg.ffn_hidden_dim();
    let wm        = WeightMap::from_file(weights_path)?;
    let n_heads   = wm.infer_n_heads(cfg.head_dim)?;

    let mut enc = EncoderTransformer::new(
        cfg.input_dim, cfg.encoder_output_dim, cfg.dim,
        cfg.n_layers, cfg.head_dim, n_heads, n_heads,
        hidden_dim, cfg.norm_eps, cfg.encoder_latent_downsample_factor, device,
    );
    load_encoder_from_wm(&wm, &mut enc, device)?;
    Ok((enc, n_heads))
}

/// Load only the **decoder** weights from a safetensors file.
///
/// Returns `(decoder, n_heads)`.
pub fn load_decoder_weights<B: Backend>(
    cfg:          &ModelConfig,
    weights_path: &str,
    device:       &B::Device,
) -> anyhow::Result<(DecoderTransformer<B>, usize)> {
    let hidden_dim = cfg.ffn_hidden_dim();
    let wm        = WeightMap::from_file(weights_path)?;
    let n_heads   = wm.infer_n_heads(cfg.head_dim)?;

    let mut dec = DecoderTransformer::new(
        cfg.input_dim, cfg.encoder_output_dim, cfg.dim, cfg.t_dim,
        cfg.n_layers, cfg.head_dim, n_heads, n_heads,
        hidden_dim, cfg.norm_eps, device,
    );
    load_decoder_from_wm(&wm, &mut dec, device)?;
    Ok((dec, n_heads))
}

// ── Full model loader ─────────────────────────────────────────────────────────

pub fn load_model<B: Backend>(
    cfg:          &ModelConfig,
    weights_path: &str,
    device:       &B::Device,
) -> anyhow::Result<EncoderDecoder<B>> {
    let hidden_dim = cfg.ffn_hidden_dim();
    let wm        = WeightMap::from_file(weights_path)?;
    let n_heads   = wm.infer_n_heads(cfg.head_dim)?;
    println!("Detected n_heads = {n_heads}");

    let mut model = EncoderDecoder::new(
        cfg.input_dim, cfg.encoder_output_dim, cfg.dim, cfg.t_dim,
        cfg.n_layers, cfg.head_dim, n_heads, n_heads,
        hidden_dim, cfg.norm_eps, cfg.encoder_latent_downsample_factor,
        cfg.stft_global_sigma as f32, device,
    );

    load_encoder_from_wm(&wm, &mut model.encoder, device)?;
    load_decoder_from_wm(&wm, &mut model.decoder, device)?;

    println!("Loaded {} weight tensors.", wm.tensors.len());
    Ok(model)
}
