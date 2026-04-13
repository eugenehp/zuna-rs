//! [`ZunaInference<B>`] — single entry point for the ZUNA EEG model.
//!
//! This is the **only** place that ties together:
//! - model architecture (`model/`)
//! - pretrained weights (`weights.rs`)
//! - preprocessing config (`config.rs`)
//! - input loading (`data.rs`)
//!
//! The binary (`bin/infer.rs`) and any downstream code use only this API.
//!
//! # Quick start
//! ```rust,ignore
//! let (zuna, ms_load) = ZunaInference::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! let result = zuna.run_fif(Path::new("recording.fif"), steps, cfg, data_norm)?;
//! result.save_safetensors("output.safetensors")?;
//! ```

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::{DataConfig, ModelConfig},
    data::{load_batch, load_from_fif, invert_reshape, FifInfo, InputBatch},
    encoder::{EncodingResult, EpochEmbedding},
    model::{encoder_decoder::EncoderDecoder, rope::RotaryEmbedding},
    weights::load_model,
};

// ── Output types ──────────────────────────────────────────────────────────────

/// Reconstructed output for one 5-second epoch.
pub struct EpochOutput {
    /// Signal: `[n_channels, n_timesteps]` row-major f32.
    pub reconstructed: Vec<f32>,
    /// Shape: `[n_channels, n_timesteps]`.
    pub shape: Vec<usize>,
    /// Electrode positions: `[n_channels, 3]` row-major f32.
    pub chan_pos: Vec<f32>,
    /// Number of EEG channels.
    pub n_channels: usize,
}

/// Result returned by [`ZunaInference::run_fif`] or [`ZunaInference::run_safetensors_batch`].
pub struct InferenceResult {
    /// One entry per 5-second epoch.
    pub epochs: Vec<EpochOutput>,
    /// Metadata from the FIF file; `None` for safetensors batch input.
    pub fif_info: Option<FifInfo>,
    /// Preprocessing time (ms).
    pub ms_preproc: f64,
    /// Model inference time, all epochs combined (ms).
    pub ms_infer: f64,
}

impl InferenceResult {
    /// Write to a safetensors file.
    ///
    /// Keys:
    /// - `reconstructed_N` — `[C, T]` float32
    /// - `chan_pos_N`       — `[C, 3]` float32
    /// - `n_samples`        — scalar float32 = number of epochs
    pub fn save_safetensors(&self, path: &str) -> anyhow::Result<()> {
        use safetensors::{Dtype, View};
        use std::borrow::Cow;

        struct F32Tensor { data: Vec<u8>, shape: Vec<usize> }
        impl View for F32Tensor {
            fn dtype(&self)    -> Dtype          { Dtype::F32 }
            fn shape(&self)    -> &[usize]        { &self.shape }
            fn data(&self)     -> Cow<'_, [u8]>   { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize            { self.data.len() }
        }
        fn to_bytes(v: &[f32]) -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        }

        // Collect owned strings + owned tensors side-by-side.
        let mut keys:    Vec<String>    = Vec::new();
        let mut tensors: Vec<F32Tensor> = Vec::new();

        for (i, ep) in self.epochs.iter().enumerate() {
            keys.push(format!("reconstructed_{i}"));
            tensors.push(F32Tensor { data: to_bytes(&ep.reconstructed), shape: ep.shape.clone() });
            keys.push(format!("chan_pos_{i}"));
            tensors.push(F32Tensor { data: to_bytes(&ep.chan_pos), shape: vec![ep.n_channels, 3] });
        }
        let n = self.epochs.len() as f32;
        keys.push("n_samples".into());
        tensors.push(F32Tensor { data: to_bytes(&[n]), shape: vec![1] });

        // safetensors::serialize wants (&str, impl View) — we give it (&str, F32Tensor).
        let pairs: Vec<(&str, F32Tensor)> =
            keys.iter().map(|s| s.as_str()).zip(tensors).collect();
        let bytes = safetensors::serialize(pairs, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ── ZunaInference ─────────────────────────────────────────────────────────────

/// ZUNA EEG Foundation Model — inference entry point.
///
/// Encapsulates architecture, weights, RoPE embeddings, and configs.
/// One instance per device; reuse across multiple recordings.
///
/// # Backend
/// Choose at compile time:
/// - CPU: `--features ndarray`  (default; + `blas-accelerate` on macOS)
/// - GPU: `--no-default-features --features wgpu`
pub struct ZunaInference<B: Backend> {
    model:     EncoderDecoder<B>,
    rope:      RotaryEmbedding<B>,
    /// Architecture hyperparameters.
    pub model_cfg: ModelConfig,
    /// Preprocessing / tokenisation parameters.
    pub data_cfg:  DataConfig,
    device:    B::Device,
}

impl<B: Backend> ZunaInference<B> {
    // ── Constructor ───────────────────────────────────────────────────────────

    /// Load model from a HuggingFace `config.json` and `model.safetensors`.
    ///
    /// Returns `(model, weight_load_ms)`.
    pub fn load(
        config_path:  &Path,
        weights_path: &Path,
        device:       B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        // 1. Parse config
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(hf_val["model"].clone())
            .context("parsing model config")?;

        // 2. Rotary positional embeddings (shared, precomputed once)
        let rope = RotaryEmbedding::<B>::new(
            model_cfg.head_dim,
            model_cfg.rope_dim,
            model_cfg.max_seqlen,
            model_cfg.rope_theta,
            &device,
        );

        // 3. Load weights
        let t = Instant::now();
        let model = load_model::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms_weights = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self { model, rope, model_cfg, data_cfg: DataConfig::default(), device }, ms_weights))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// One-line description of the loaded model.
    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        // n_heads is inferred from weights (actual); approximation here is dim/head_dim
        format!(
            "ZUNA  dim={}  layers={}  head_dim={}  ffn_hidden={}  \
             rope_dim={}  max_seqlen={}",
            c.dim, c.n_layers, c.head_dim, c.ffn_hidden_dim(), c.rope_dim, c.max_seqlen,
        )
    }

    // ── Inference from a raw FIF recording ────────────────────────────────────

    /// Full pure-Rust pipeline: `.fif` → exg → ZUNA model.
    ///
    /// # Arguments
    /// - `steps`     — diffusion denoising steps (50 → full quality; 10 → fast preview)
    /// - `cfg`       — classifier-free guidance scale (1.0 = disabled)
    /// - `data_norm` — divisor applied to z-scored signal before entering model;
    ///                 same value is multiplied back into the output
    pub fn run_fif(
        &self,
        fif_path:  &Path,
        steps:     usize,
        cfg:       f32,
        data_norm: f32,
    ) -> anyhow::Result<InferenceResult> {
        let t_pp = Instant::now();
        let (batches, fif_info) = load_from_fif::<B>(
            fif_path, &self.data_cfg, data_norm, &self.device,
        ).with_context(|| format!("exg on {}", fif_path.display()))?;
        let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;

        let t_inf = Instant::now();
        let epochs = self.run_batches(batches, steps, cfg, data_norm)?;
        let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceResult { epochs, fif_info: Some(fif_info), ms_preproc, ms_infer })
    }

    /// Input from a pre-processed safetensors batch (Python / legacy path).
    pub fn run_safetensors_batch(
        &self,
        batch_path: &Path,
        steps:      usize,
        cfg:        f32,
        data_norm:  f32,
    ) -> anyhow::Result<InferenceResult> {
        let t_pp = Instant::now();
        let batches = load_batch::<B>(
            batch_path.to_str().context("batch path not valid UTF-8")?,
            &self.data_cfg,
            &self.device,
        )?;
        let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;

        let t_inf = Instant::now();
        let epochs = self.run_batches(batches, steps, cfg, data_norm)?;
        let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceResult { epochs, fif_info: None, ms_preproc, ms_infer })
    }

    // ── Encode-only convenience methods ───────────────────────────────────────

    /// Preprocess a `.fif` recording and encode it into latent embeddings,
    /// **without** running the decoder.
    ///
    /// Equivalent to [`crate::encoder::ZunaEncoder::encode_fif`] but uses the
    /// encoder that is already loaded as part of the full model — no extra
    /// weight loading required.
    pub fn encode_fif(
        &self,
        fif_path:  &Path,
        data_norm: f32,
    ) -> anyhow::Result<EncodingResult> {
        let t_pp = Instant::now();
        let (batches, fif_info) = load_from_fif::<B>(
            fif_path, &self.data_cfg, data_norm, &self.device,
        ).with_context(|| format!("exg on {}", fif_path.display()))?;
        let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;

        let t_enc = Instant::now();
        let epochs = self.encode_inputs(batches)?;
        let ms_encode = t_enc.elapsed().as_secs_f64() * 1000.0;

        Ok(EncodingResult { epochs, fif_info: Some(fif_info), ms_preproc, ms_encode })
    }

    /// Encode a pre-processed safetensors batch into latent embeddings,
    /// **without** running the decoder.
    pub fn encode_batch(
        &self,
        batch_path: &Path,
    ) -> anyhow::Result<EncodingResult> {
        let t_pp = Instant::now();
        let batches = load_batch::<B>(
            batch_path.to_str().context("batch path not valid UTF-8")?,
            &self.data_cfg,
            &self.device,
        )?;
        let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;

        let t_enc = Instant::now();
        let epochs = self.encode_inputs(batches)?;
        let ms_encode = t_enc.elapsed().as_secs_f64() * 1000.0;

        Ok(EncodingResult { epochs, fif_info: None, ms_preproc, ms_encode })
    }

    fn encode_inputs(
        &self,
        batches: Vec<InputBatch<B>>,
    ) -> anyhow::Result<Vec<EpochEmbedding>> {
        batches.into_iter().map(|batch| {
            let n_channels    = batch.n_channels;
            let tc            = batch.tc;
            let tok_idx_saved = batch.tok_idx.clone();
            let chan_pos_saved = batch.chan_pos.clone();

            // Encoder forward: [1, S, output_dim]
            let enc_out = self.model.encoder.forward(
                batch.encoder_input,
                batch.tok_idx,
                &self.rope,
            );
            let [_, s, output_dim] = enc_out.dims();

            let embeddings = enc_out
                .squeeze::<2>()
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("embedding→vec: {e:?}"))?;

            // NdArray backend stores Int as i64; wgpu backend stores Int as i32.
            let tok_idx_data = tok_idx_saved.into_data();
            let tok_idx: Vec<i64> = tok_idx_data
                .to_vec::<i64>()
                .or_else(|_| tok_idx_data.to_vec::<i32>()
                    .map(|v| v.into_iter().map(|x| x as i64).collect()))
                .map_err(|e| anyhow::anyhow!("tok_idx→vec: {e:?}"))?;

            let chan_pos = chan_pos_saved
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("chan_pos→vec: {e:?}"))?;

            Ok(EpochEmbedding {
                embeddings,
                shape: vec![s, output_dim],
                tok_idx,
                chan_pos,
                n_channels,
                tc,
            })
        }).collect()
    }

    // ── Decode from pre-computed embeddings ──────────────────────────────────

    /// Decode [`EncodingResult`] through the decoder diffusion loop.
    ///
    /// Lets you run encode and decode as separate timed steps.
    /// Each epoch is decoded independently; returned `ms_infer` covers all epochs.
    pub fn decode_embeddings(
        &self,
        embeddings: &EncodingResult,
        steps:      usize,
        cfg:        f32,
        data_norm:  f32,
    ) -> anyhow::Result<InferenceResult> {
        let t = Instant::now();
        let epochs = embeddings.epochs
            .iter()
            .map(|ep| self.decode_epoch(ep, steps, cfg, data_norm))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ms_infer = t.elapsed().as_secs_f64() * 1000.0;
        Ok(InferenceResult { epochs, fif_info: None, ms_preproc: 0.0, ms_infer })
    }

    /// Decode a **single** [`EpochEmbedding`] through the decoder diffusion loop.
    ///
    /// Use this to time each epoch individually.
    pub fn decode_epoch(
        &self,
        ep:        &EpochEmbedding,
        steps:     usize,
        cfg:       f32,
        data_norm: f32,
    ) -> anyhow::Result<EpochOutput> {
        use burn::tensor::Distribution;

        let n_tokens = ep.n_tokens();
        let dc       = &self.data_cfg;

        // Reconstruct encoder conditioning from stored Vec<f32>.
        let enc_out = Tensor::<B, 2>::from_data(
            TensorData::new(ep.embeddings.clone(), ep.shape.clone()),
            &self.device,
        )
        .unsqueeze_dim::<3>(0); // [1, S, output_dim]

        let tok_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ep.tok_idx.clone(), vec![n_tokens, 4]),
            &self.device,
        );

        let [b, s, d] = enc_out.dims();
        let dt = 1.0_f32 / steps as f32;

        // Initial noise z ~ N(0, σ²)
        let mut z = Tensor::<B, 3>::random(
            [b, s, d],
            Distribution::Normal(0.0, self.model.global_sigma as f64),
            &self.device,
        );

        // Rectified-flow Euler diffusion loop
        for i in (1..=steps).rev() {
            let t_val  = dt * i as f32;
            let time_t = Tensor::<B, 3>::full([b, 1, 1], t_val, &self.device);

            let vc = self.model.decoder.forward(
                z.clone(), enc_out.clone(), time_t.clone(), tok_idx.clone(), &self.rope,
            );
            let vc = if (cfg - 1.0).abs() > 1e-4 {
                let enc_zeros = Tensor::zeros([b, s, d], &self.device);
                let vc_u = self.model.decoder.forward(
                    z.clone(), enc_zeros, time_t, tok_idx.clone(), &self.rope,
                );
                vc_u.clone() + (vc - vc_u).mul_scalar(cfg)
            } else {
                vc
            };
            z = z - vc.mul_scalar(dt);
        }

        let [_, s, tf] = z.dims();
        let recon = invert_reshape(
            z.reshape([s, tf]), ep.n_channels, ep.tc, dc.num_fine_time_pts,
        );
        let recon = recon.mul_scalar(data_norm);
        let shape         = recon.dims().to_vec();
        let reconstructed = recon.into_data().to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("recon→vec: {e:?}"))?;
        Ok(EpochOutput { reconstructed, shape, chan_pos: ep.chan_pos.clone(), n_channels: ep.n_channels })
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn run_batches(
        &self,
        batches:   Vec<InputBatch<B>>,
        steps:     usize,
        cfg:       f32,
        data_norm: f32,
    ) -> anyhow::Result<Vec<EpochOutput>> {
        let dc = &self.data_cfg;
        batches.into_iter().map(|batch| {
            let z = self.model.sample(
                batch.encoder_input,
                batch.tok_idx,
                &self.rope,
                steps,
                cfg,
            );
            let [_, s, tf] = z.dims();
            let z     = z.reshape([s, tf]);
            let recon = invert_reshape(z, batch.n_channels, batch.tc, dc.num_fine_time_pts);
            let recon = recon.mul_scalar(data_norm);

            let shape         = recon.dims().to_vec();
            let reconstructed = recon.into_data().to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("tensor→vec: {e:?}"))?;
            let chan_pos = batch.chan_pos.into_data().to_vec::<f32>()
                .map_err(|e| anyhow::anyhow!("chan_pos→vec: {e:?}"))?;

            Ok(EpochOutput { reconstructed, shape, chan_pos, n_channels: batch.n_channels })
        }).collect()
    }
}
