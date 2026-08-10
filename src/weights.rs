//! Load pretrained ZUNA weights from a safetensors file (burn 0.20.1)
//!
//! The HuggingFace file stores weights as bfloat16 under keys like
//!   `model.encoder.tok_embeddings.weight`
//!   `model.decoder.layers.0.cross_attention.wq.weight`
//!
//! We strip the leading `model.` prefix and convert bf16 → f32.
//!
//! ## Weight key reference
//!
//! ```text
//! ENCODER
//!   encoder.tok_embeddings.weight            [1024, 32]  no bias
//!   encoder.registers                        [1, 32]     (Param, not Linear)
//!   encoder.norm.inner.gamma                 [1024]      (RmsNorm)
//!   encoder.output.weight                    [32, 1024]  no bias
//!   encoder.layers.{i}.attention_norm.inner.gamma        [1024]
//!   encoder.layers.{i}.attention.wq.weight               [1024,1024]
//!   encoder.layers.{i}.attention.wk.weight               [1024,1024]
//!   encoder.layers.{i}.attention.wv.weight               [1024,1024]
//!   encoder.layers.{i}.attention.wo.weight               [1024,1024]
//!   encoder.layers.{i}.ffn_norm.inner.gamma              [1024]
//!   encoder.layers.{i}.feed_forward.w1.weight            [2816,1024]
//!   encoder.layers.{i}.feed_forward.w2.weight            [1024,2816]
//!   encoder.layers.{i}.feed_forward.w3.weight            [2816,1024]
//!
//! DECODER
//!   decoder.tok_embeddings.weight            [1024, 32]  no bias
//!   decoder.t_embedder.weight                [32, 1]     (buffer)
//!   decoder.t_embedder.proj.weight           [64, 64]    + bias [64]
//!   decoder.encoder_proj.weight              [1024, 32]  + bias [1024]
//!   decoder.norm.weight.weight               [1024, 64]  + bias [1024]   (AdaRMSNorm)
//!   decoder.output.weight                    [32, 1024]  no bias
//!   decoder.layers.{i}.cross_attention_x_norm.weight.weight  [1024,64]  + bias
//!   decoder.layers.{i}.cross_attention_y_norm.weight.weight  [1024,64]  + bias
//!   decoder.layers.{i}.cross_attention.wq/wk/wv/wo.weight    [1024,1024]
//!   decoder.layers.{i}.attention_norm.weight.weight           [1024,64]  + bias
//!   decoder.layers.{i}.attention.wq/wk/wv/wo.weight          [1024,1024]
//!   decoder.layers.{i}.ffn_norm.weight.weight                 [1024,64]  + bias
//!   decoder.layers.{i}.feed_forward.w1/w2/w3.weight          various
//! ```
//!
//! ## ZUNA1.1 additions
//!
//! ZUNA1.1 nests every plain RMSNorm one level deeper (`…ffn_norm.norm.weight`);
//! [`canonical_key`] collapses that back to the ZUNA1 spelling on load, so the
//! keys above are version-independent. On top of those, ZUNA1.1 adds:
//!
//! ```text
//!   {enc,dec}.layers.{i}.attention.q_norm.weight             [64]   (QK-norm)
//!   {enc,dec}.layers.{i}.attention.k_norm.weight             [64]
//!   decoder.layers.{i}.cross_attention.{q,k}_norm.weight     [64]
//!   {enc,dec}.layers.{i}.attention_norm_post.weight          [1024] (sandwich)
//!   {enc,dec}.layers.{i}.ffn_norm_post.weight                [1024]
//!   decoder.layers.{i}.cross_attention_norm_post.weight      [1024]
//! ```

use burn::module::{Param, ParamId};
use burn::prelude::*;
use half::bf16;
use safetensors::SafeTensors;
use std::collections::HashMap;

use crate::config::{canonical_key, ModelArch, ModelConfig};
use crate::model::decoder::DecoderTransformer;
use crate::model::encoder::EncoderTransformer;
use crate::model::encoder_decoder::EncoderDecoder;
use crate::model::norm::RMSNorm;
use crate::model::{DecoderDims, EncoderDims};

// ── Raw tensor map ────────────────────────────────────────────────────────────

pub struct WeightMap {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

/// Optional prefix filter for [`WeightMap::from_file_filtered`].
///
/// When set to `Encoder`, only tensors whose (prefix-stripped) key starts with
/// `"encoder."` or `"encoder_"` are loaded.  All others are skipped — saving
/// roughly half the bf16→f32 conversion work and memory when only embeddings
/// are needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFilter {
    /// Load every tensor in the file.
    All,
    /// Load only encoder tensors (skip decoder).
    Encoder,
}

impl WeightMap {
    /// Load all tensors from a safetensors file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        Self::from_file_filtered(path, WeightFilter::All)
    }

    /// Load tensors from a safetensors file, optionally filtering by component.
    ///
    /// When `filter` is [`WeightFilter::Encoder`], decoder tensors are skipped
    /// entirely — no bf16→f32 conversion or allocation is performed for them.
    /// This roughly halves load time and peak memory for encoder-only use.
    pub fn from_file_filtered(path: &str, filter: WeightFilter) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let n_tensors = st.len();
        let mut tensors = HashMap::with_capacity(n_tensors);

        for (raw_key, view) in st.tensors() {
            let stripped = raw_key.strip_prefix("model.").unwrap_or(raw_key.as_str());
            let key = canonical_key(stripped).into_owned();

            // Skip tensors that don't match the filter.
            if filter == WeightFilter::Encoder
                && !key.starts_with("encoder.")
                && !key.starts_with("encoder_")
            {
                continue;
            }

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

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

    /// Take a tensor by key, **removing** it from the map to avoid cloning.
    ///
    /// Prefer this over [`Self::get`] when you are loading weights into a model
    /// and won't need the same key again — it avoids a full `Vec<f32>` clone.
    pub fn take<B: Backend, const N: usize>(
        &mut self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self
            .tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data, shape),
            device,
        ))
    }

    /// Load a tensor by its (prefix-stripped) safetensors key.
    ///
    /// **Note:** this clones the underlying `Vec<f32>`.  Prefer [`Self::take`]
    /// when each key is consumed exactly once (typical weight-loading pattern).
    pub fn get<B: Backend, const N: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> anyhow::Result<Tensor<B, N>> {
        let (data, shape) = self
            .tensors
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("weight key not found: {key}"))?;

        if shape.len() != N {
            anyhow::bail!("rank mismatch for {key}: expected {N}, got {}", shape.len());
        }

        Ok(Tensor::<B, N>::from_data(
            TensorData::new(data.clone(), shape.clone()),
            device,
        ))
    }

    /// Detect which ZUNA architecture this checkpoint holds, from the
    /// (canonicalised) tensor names.
    pub fn detect_arch(&self) -> ModelArch {
        ModelArch::detect(|k| self.tensors.contains_key(k))
    }

    /// Infer the number of attention heads from the wq weight shape.
    /// wq is stored as [n_heads * head_dim, dim] (PyTorch out×in convention).
    pub fn infer_n_heads(&self, head_dim: usize) -> anyhow::Result<usize> {
        anyhow::ensure!(head_dim > 0, "head_dim must be > 0");
        let key = "encoder.layers.0.attention.wq.weight";
        let (_, shape) = self
            .tensors
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("key not found for n_heads inference: {key}"))?;
        anyhow::ensure!(
            shape.len() >= 2,
            "wq weight must be 2-D, got shape {shape:?}"
        );
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
    w: Tensor<B, 2>, // raw safetensors tensor [out, in]
) {
    // Transpose [out, in] → [in, out] for burn's Row layout.
    // Construct a fresh Param directly — avoids triggering lazy random init
    // that burn's `.clone().map()` pattern would cause.
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
}

/// Assign weight + bias into a `burn::nn::Linear`.
fn set_linear_wb<B: Backend>(
    linear: &mut burn::nn::Linear<B>,
    w: Tensor<B, 2>, // raw safetensors tensor [out, in]
    b: Tensor<B, 1>,
) {
    linear.weight = Param::initialized(ParamId::new(), w.transpose());
    linear.bias = Some(Param::initialized(ParamId::new(), b));
}

/// Assign a 1-D weight into a `burn::nn::RmsNorm` (field: `gamma`).
fn set_rmsnorm<B: Backend>(norm: &mut burn::nn::RmsNorm<B>, w: Tensor<B, 1>) {
    norm.gamma = Param::initialized(ParamId::new(), w);
}

/// Fill an optional [`RMSNorm`] (QK-norm or sandwich norm) from `key`.
///
/// The `Option` is `Some` exactly when [`ModelArch`] detection said the
/// checkpoint has this norm, so a missing tensor here is a real error.
fn set_opt_rmsnorm<B: Backend>(
    wm: &mut WeightMap,
    slot: &mut Option<RMSNorm<B>>,
    key: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    if let Some(norm) = slot.as_mut() {
        let w: Tensor<B, 1> = wm.take(key, device)?;
        set_rmsnorm(&mut norm.inner, w);
    }
    Ok(())
}

/// Fill an attention module's optional QK-norms from `<prefix>.{q,k}_norm.weight`.
fn set_qk_norms<B: Backend>(
    wm: &mut WeightMap,
    q_norm: &mut Option<RMSNorm<B>>,
    k_norm: &mut Option<RMSNorm<B>>,
    prefix: &str,
    device: &B::Device,
) -> anyhow::Result<()> {
    set_opt_rmsnorm(wm, q_norm, &format!("{prefix}.q_norm.weight"), device)?;
    set_opt_rmsnorm(wm, k_norm, &format!("{prefix}.k_norm.weight"), device)?;
    Ok(())
}

// ── Internal per-component loaders ───────────────────────────────────────────

/// Populate an already-constructed [`EncoderTransformer`] from a [`WeightMap`].
///
/// Uses [`WeightMap::take`] to move tensor data out of the map without cloning.
fn load_encoder_from_wm<B: Backend>(
    wm: &mut WeightMap,
    enc: &mut EncoderTransformer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    set_linear_wb(
        &mut enc.tok_embeddings,
        wm.take("encoder.tok_embeddings.weight", device)?,
        wm.take("encoder.tok_embeddings.bias", device)?,
    );

    let regs: Tensor<B, 2> = wm.take("encoder.registers", device)?;
    enc.registers = Param::initialized(ParamId::new(), regs);

    let norm_w: Tensor<B, 1> = wm.take("encoder.norm.weight", device)?;
    set_rmsnorm(&mut enc.norm.inner, norm_w);

    set_linear_w(&mut enc.output, wm.take("encoder.output.weight", device)?);

    for (i, layer) in enc.layers.iter_mut().enumerate() {
        let p = format!("encoder.layers.{i}");

        let an_w: Tensor<B, 1> = wm.take(&format!("{p}.attention_norm.weight"), device)?;
        set_rmsnorm(&mut layer.attention_norm.inner, an_w);

        set_linear_w(
            &mut layer.attention.wq,
            wm.take(&format!("{p}.attention.wq.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wk,
            wm.take(&format!("{p}.attention.wk.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wv,
            wm.take(&format!("{p}.attention.wv.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wo,
            wm.take(&format!("{p}.attention.wo.weight"), device)?,
        );
        set_qk_norms(
            wm,
            &mut layer.attention.q_norm,
            &mut layer.attention.k_norm,
            &format!("{p}.attention"),
            device,
        )?;
        set_opt_rmsnorm(
            wm,
            &mut layer.attention_norm_post,
            &format!("{p}.attention_norm_post.weight"),
            device,
        )?;

        let fn_w: Tensor<B, 1> = wm.take(&format!("{p}.ffn_norm.weight"), device)?;
        set_rmsnorm(&mut layer.ffn_norm.inner, fn_w);

        set_linear_w(
            &mut layer.feed_forward.w1,
            wm.take(&format!("{p}.feed_forward.w1.weight"), device)?,
        );
        set_linear_w(
            &mut layer.feed_forward.w2,
            wm.take(&format!("{p}.feed_forward.w2.weight"), device)?,
        );
        set_linear_w(
            &mut layer.feed_forward.w3,
            wm.take(&format!("{p}.feed_forward.w3.weight"), device)?,
        );
        set_opt_rmsnorm(
            wm,
            &mut layer.ffn_norm_post,
            &format!("{p}.ffn_norm_post.weight"),
            device,
        )?;
    }
    Ok(())
}

/// Populate an already-constructed [`DecoderTransformer`] from a [`WeightMap`].
///
/// Uses [`WeightMap::take`] to move tensor data out of the map without cloning.
fn load_decoder_from_wm<B: Backend>(
    wm: &mut WeightMap,
    dec: &mut DecoderTransformer<B>,
    device: &B::Device,
) -> anyhow::Result<()> {
    set_linear_wb(
        &mut dec.tok_embeddings,
        wm.take("decoder.tok_embeddings.weight", device)?,
        wm.take("decoder.tok_embeddings.bias", device)?,
    );

    let fc_w: Tensor<B, 2> = wm.take("decoder.t_embedder.weight", device)?;
    dec.t_embedder.weight = Param::initialized(ParamId::new(), fc_w);
    set_linear_wb(
        &mut dec.t_embedder.proj,
        wm.take("decoder.t_embedder.proj.weight", device)?,
        wm.take("decoder.t_embedder.proj.bias", device)?,
    );

    set_linear_wb(
        &mut dec.encoder_proj,
        wm.take("decoder.encoder_proj.weight", device)?,
        wm.take("decoder.encoder_proj.bias", device)?,
    );

    // AdaRMSNorm final norm — inner Linear is named "weight" in PyTorch.
    // safetensors keys: decoder.norm.weight.weight / decoder.norm.weight.bias
    set_linear_wb(
        &mut dec.norm.weight,
        wm.take("decoder.norm.weight.weight", device)?,
        wm.take("decoder.norm.weight.bias", device)?,
    );

    set_linear_w(&mut dec.output, wm.take("decoder.output.weight", device)?);

    for (i, layer) in dec.layers.iter_mut().enumerate() {
        let p = format!("decoder.layers.{i}");

        set_linear_wb(
            &mut layer.cross_attention_x_norm.weight,
            wm.take(&format!("{p}.cross_attention_x_norm.weight.weight"), device)?,
            wm.take(&format!("{p}.cross_attention_x_norm.weight.bias"), device)?,
        );
        set_linear_wb(
            &mut layer.cross_attention_y_norm.weight,
            wm.take(&format!("{p}.cross_attention_y_norm.weight.weight"), device)?,
            wm.take(&format!("{p}.cross_attention_y_norm.weight.bias"), device)?,
        );

        set_linear_w(
            &mut layer.cross_attention.wq,
            wm.take(&format!("{p}.cross_attention.wq.weight"), device)?,
        );
        set_linear_w(
            &mut layer.cross_attention.wk,
            wm.take(&format!("{p}.cross_attention.wk.weight"), device)?,
        );
        set_linear_w(
            &mut layer.cross_attention.wv,
            wm.take(&format!("{p}.cross_attention.wv.weight"), device)?,
        );
        set_linear_w(
            &mut layer.cross_attention.wo,
            wm.take(&format!("{p}.cross_attention.wo.weight"), device)?,
        );
        set_qk_norms(
            wm,
            &mut layer.cross_attention.q_norm,
            &mut layer.cross_attention.k_norm,
            &format!("{p}.cross_attention"),
            device,
        )?;
        set_opt_rmsnorm(
            wm,
            &mut layer.cross_attention_norm_post,
            &format!("{p}.cross_attention_norm_post.weight"),
            device,
        )?;

        set_linear_wb(
            &mut layer.attention_norm.weight,
            wm.take(&format!("{p}.attention_norm.weight.weight"), device)?,
            wm.take(&format!("{p}.attention_norm.weight.bias"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wq,
            wm.take(&format!("{p}.attention.wq.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wk,
            wm.take(&format!("{p}.attention.wk.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wv,
            wm.take(&format!("{p}.attention.wv.weight"), device)?,
        );
        set_linear_w(
            &mut layer.attention.wo,
            wm.take(&format!("{p}.attention.wo.weight"), device)?,
        );
        set_qk_norms(
            wm,
            &mut layer.attention.q_norm,
            &mut layer.attention.k_norm,
            &format!("{p}.attention"),
            device,
        )?;
        set_opt_rmsnorm(
            wm,
            &mut layer.attention_norm_post,
            &format!("{p}.attention_norm_post.weight"),
            device,
        )?;

        set_linear_wb(
            &mut layer.ffn_norm.weight,
            wm.take(&format!("{p}.ffn_norm.weight.weight"), device)?,
            wm.take(&format!("{p}.ffn_norm.weight.bias"), device)?,
        );
        set_linear_w(
            &mut layer.feed_forward.w1,
            wm.take(&format!("{p}.feed_forward.w1.weight"), device)?,
        );
        set_linear_w(
            &mut layer.feed_forward.w2,
            wm.take(&format!("{p}.feed_forward.w2.weight"), device)?,
        );
        set_linear_w(
            &mut layer.feed_forward.w3,
            wm.take(&format!("{p}.feed_forward.w3.weight"), device)?,
        );
        set_opt_rmsnorm(
            wm,
            &mut layer.ffn_norm_post,
            &format!("{p}.ffn_norm_post.weight"),
            device,
        )?;
    }
    Ok(())
}

// ── Public partial loaders ────────────────────────────────────────────────────

/// Load only the **encoder** weights from a safetensors file.
///
/// Only encoder tensors are deserialized from the safetensors file — decoder
/// tensors are skipped entirely, halving the bf16→f32 conversion work, peak
/// memory, and drop cost compared to loading the full model.
///
/// Tensor data is moved (not cloned) into the burn model, avoiding an extra
/// copy of every weight vector.
///
/// Returns `(encoder, n_heads, arch)`.  `n_heads` is inferred from the weight
/// shape and `arch` from the weight names, because `config.json` stores
/// neither.
pub fn load_encoder_weights<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<(EncoderTransformer<B>, usize, ModelArch)> {
    let mut wm = WeightMap::from_file_filtered(weights_path, WeightFilter::Encoder)?;
    let n_heads = wm.infer_n_heads(cfg.head_dim)?;
    let arch = wm.detect_arch();

    let mut enc = EncoderTransformer::new(EncoderDims::from_config(cfg, n_heads, arch), device);
    load_encoder_from_wm(&mut wm, &mut enc, device)?;
    Ok((enc, n_heads, arch))
}

/// Load only the **decoder** weights from a safetensors file.
///
/// Returns `(decoder, n_heads, arch)`.
pub fn load_decoder_weights<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<(DecoderTransformer<B>, usize, ModelArch)> {
    let mut wm = WeightMap::from_file(weights_path)?;
    let n_heads = wm.infer_n_heads(cfg.head_dim)?;
    let arch = wm.detect_arch();

    let mut dec = DecoderTransformer::new(DecoderDims::from_config(cfg, n_heads, arch), device);
    load_decoder_from_wm(&mut wm, &mut dec, device)?;
    Ok((dec, n_heads, arch))
}

// ── Full model loader ─────────────────────────────────────────────────────────

pub fn load_model<B: Backend>(
    cfg: &ModelConfig,
    weights_path: &str,
    device: &B::Device,
) -> anyhow::Result<EncoderDecoder<B>> {
    let mut wm = WeightMap::from_file(weights_path)?;
    let n_heads = wm.infer_n_heads(cfg.head_dim)?;
    let arch = wm.detect_arch();
    println!("Detected {} — n_heads = {n_heads}", arch.label());

    let mut model = EncoderDecoder::new(
        EncoderDims::from_config(cfg, n_heads, arch),
        DecoderDims::from_config(cfg, n_heads, arch),
        cfg.stft_global_sigma as f32,
        device,
    );

    load_encoder_from_wm(&mut wm, &mut model.encoder, device)?;
    load_decoder_from_wm(&mut wm, &mut model.decoder, device)?;

    println!("Loaded {} weight tensors.", wm.tensors.len());
    Ok(model)
}
