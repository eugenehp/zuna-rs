//! Standalone ZUNA encoder — produce latent EEG embeddings.
//!
//! # Regularisation
//!
//! The ZUNA encoder uses an **MMD bottleneck** (Maximum Mean Discrepancy).
//! During training, an MMD loss constrains the encoder's output distribution
//! to be close to **N(0, I)**.  At inference the bottleneck is a pure
//! passthrough — no reparameterisation or additional normalisation is applied.
//! The weights therefore carry the regularisation implicitly: the encoder
//! output is already approximately normally distributed with zero mean and
//! unit variance per dimension.
//!
//! # Loading — encoder only
//!
//! ```rust,ignore
//! use zuna_rs::encoder::ZunaEncoder;
//!
//! let (enc, ms) = ZunaEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//!
//! // Encode a FIF recording
//! let result = enc.encode_fif(Path::new("recording.fif"), 10.0)?;
//! result.save_safetensors("data/embeddings.safetensors")?;
//! ```

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::prelude::*;

use crate::{
    config::{DataConfig, ModelConfig},
    data::{load_batch, load_from_fif, FifInfo, InputBatch},
    model::{encoder::EncoderTransformer, rope::RotaryEmbedding},
    weights::load_encoder_weights,
};

// ── Output types ──────────────────────────────────────────────────────────────

/// Per-epoch encoder embedding produced by [`ZunaEncoder`].
///
/// Represents one 5-second EEG epoch in the model's latent space.
///
/// ## Shape
/// `embeddings` is a flat row-major `f32` buffer of shape
/// `[n_tokens, output_dim]` where:
/// - `n_tokens = n_channels × tc`  (S in model notation)
/// - `output_dim = encoder_output_dim` from config (32 by default)
/// - `tc = T / num_fine_time_pts`   (coarse time steps)
///
/// ## Regularisation
/// Per the MMD training objective, each dimension of the embedding is
/// approximately N(0, 1) at the population level.  Individual samples will
/// deviate; no further normalisation is needed before downstream use.
pub struct EpochEmbedding {
    /// Latent tokens: row-major f32, shape `[n_tokens, output_dim]`.
    pub embeddings: Vec<f32>,
    /// Shape `[n_tokens, output_dim]`.
    pub shape: Vec<usize>,
    /// Discrete token indices needed to re-decode this embedding.
    /// Row-major i64, shape `[n_tokens, 4]`  (x_bin, y_bin, z_bin, t_coarse).
    pub tok_idx: Vec<i64>,
    /// Channel positions in metres, row-major f32, shape `[n_channels, 3]`.
    pub chan_pos: Vec<f32>,
    /// Number of EEG channels (C).
    pub n_channels: usize,
    /// Coarse time steps per epoch (tc = T / num_fine_time_pts).
    pub tc: usize,
}

impl EpochEmbedding {
    /// Total number of tokens  S = n_channels × tc.
    #[inline] pub fn n_tokens(&self) -> usize { self.n_channels * self.tc }
    /// Output dimension of the encoder bottleneck (32 by default).
    #[inline] pub fn output_dim(&self) -> usize { self.shape.get(1).copied().unwrap_or(0) }
}

/// Collection of per-epoch embeddings returned by [`ZunaEncoder`].
pub struct EncodingResult {
    /// One entry per 5-second EEG epoch.
    pub epochs: Vec<EpochEmbedding>,
    /// Metadata extracted from the FIF file; `None` for safetensors batch input.
    pub fif_info: Option<FifInfo>,
    /// Preprocessing time in milliseconds.
    pub ms_preproc: f64,
    /// Encoder forward-pass time in milliseconds (all epochs combined).
    pub ms_encode: f64,
}

impl EncodingResult {
    /// Persist embeddings to a safetensors file.
    ///
    /// Keys written per epoch `N`:
    /// - `embeddings_N` — `[n_tokens, output_dim]` float32
    /// - `tok_idx_N`    — `[n_tokens, 4]` int32  (needed for decoding)
    /// - `chan_pos_N`   — `[n_channels, 3]` float32
    ///
    /// Plus:
    /// - `n_samples`    — scalar float32 = number of epochs
    pub fn save_safetensors(&self, path: &str) -> anyhow::Result<()> {
        use safetensors::{Dtype, View};
        use std::borrow::Cow;

        struct RawTensor { data: Vec<u8>, shape: Vec<usize>, dtype: Dtype }
        impl View for RawTensor {
            fn dtype(&self)    -> Dtype         { self.dtype }
            fn shape(&self)    -> &[usize]      { &self.shape }
            fn data(&self)     -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize          { self.data.len() }
        }

        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let i64_bytes = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|i| i.to_le_bytes()).collect() };

        let mut keys:    Vec<String>    = Vec::new();
        let mut tensors: Vec<RawTensor> = Vec::new();

        for (i, ep) in self.epochs.iter().enumerate() {
            let n_tok = ep.n_tokens();

            keys.push(format!("embeddings_{i}"));
            tensors.push(RawTensor {
                data: f32_bytes(&ep.embeddings),
                shape: ep.shape.clone(),
                dtype: Dtype::F32,
            });

            keys.push(format!("tok_idx_{i}"));
            tensors.push(RawTensor {
                data: i64_bytes(&ep.tok_idx),
                shape: vec![n_tok, 4],
                dtype: Dtype::I64,
            });

            keys.push(format!("chan_pos_{i}"));
            tensors.push(RawTensor {
                data: f32_bytes(&ep.chan_pos),
                shape: vec![ep.n_channels, 3],
                dtype: Dtype::F32,
            });
        }

        let n = self.epochs.len() as f32;
        keys.push("n_samples".into());
        tensors.push(RawTensor { data: f32_bytes(&[n]), shape: vec![1], dtype: Dtype::F32 });

        let pairs: Vec<(&str, RawTensor)> = keys.iter().map(|s| s.as_str()).zip(tensors).collect();
        let bytes = safetensors::serialize(pairs, None)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ── ZunaEncoder ───────────────────────────────────────────────────────────────

/// Standalone ZUNA encoder.
///
/// Loads only the encoder half of the pretrained weights — useful when you only
/// need latent embeddings and want to save memory and startup time compared to
/// loading the full [`crate::ZunaInference`].
///
/// # Backend
/// Compile-time choice (same as the full model):
/// - CPU (default): `--features ndarray`
/// - GPU: `--no-default-features --features wgpu`
pub struct ZunaEncoder<B: Backend> {
    encoder:       EncoderTransformer<B>,
    rope:          RotaryEmbedding<B>,
    /// Architecture hyperparameters (from config.json).
    pub model_cfg: ModelConfig,
    /// Preprocessing / tokenisation parameters.
    pub data_cfg:  DataConfig,
    device:        B::Device,
}

impl<B: Backend> ZunaEncoder<B> {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Load encoder weights from a HuggingFace `config.json` and
    /// `model.safetensors`.  Decoder tensors are read from disk but not kept
    /// in memory (the full file is parsed once for key extraction).
    ///
    /// Returns `(encoder, weight_load_ms)`.
    pub fn load(
        config_path:  &Path,
        weights_path: &Path,
        device:       B::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(hf_val["model"].clone())
            .context("parsing model config")?;

        let rope = RotaryEmbedding::<B>::new(
            model_cfg.head_dim, model_cfg.rope_dim,
            model_cfg.max_seqlen, model_cfg.rope_theta, &device,
        );

        let t = Instant::now();
        let (encoder, n_heads) = load_encoder_weights::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("Detected n_heads = {n_heads}");

        Ok((Self { encoder, rope, model_cfg, data_cfg: DataConfig::default(), device }, ms))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// One-line description of the loaded encoder.
    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "ZUNA encoder  dim={}  layers={}  head_dim={}  out_dim={}",
            c.dim, c.n_layers, c.head_dim, c.encoder_output_dim,
        )
    }

    // ── High-level encode API ─────────────────────────────────────────────────

    /// Preprocess a `.fif` recording and encode it into latent embeddings.
    ///
    /// `data_norm` is the same divisor used to train ZUNA (default: 10.0).
    /// It is applied during preprocessing; the encoder output is **not**
    /// re-scaled — it reflects the MMD-regularised latent space directly.
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

    /// Encode a pre-processed safetensors batch (Python / legacy input path).
    ///
    /// The batch is assumed to already be normalised (÷ data_norm); the
    /// `data_norm` argument is **not** applied again here — it exists only to
    /// document the convention used when the file was created.
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

    /// Encode a single prepared [`InputBatch`], returning the raw encoder
    /// output tensor `[1, S, output_dim]`.
    ///
    /// This is the **MMD-regularised embedding**: training constrains the
    /// distribution to N(0, I); at inference the bottleneck is a passthrough.
    /// No further normalisation is applied here.
    pub fn encode_tensor(&self, batch: &InputBatch<B>) -> Tensor<B, 3> {
        self.encoder.forward(
            batch.encoder_input.clone(),
            batch.tok_idx.clone(),
            &self.rope,
        )
    }

    // ── Lower-level API (benchmark / export) ─────────────────────────────────

    /// Run the FIF preprocessing pipeline and return raw [`InputBatch`]es
    /// **without** running the encoder.
    ///
    /// Use together with [`Self::encode_batches`] to time encode separately,
    /// or to export the pre-tokenised tensors for external comparison.
    pub fn preprocess_fif(
        &self,
        fif_path:  &Path,
        data_norm: f32,
    ) -> anyhow::Result<(Vec<InputBatch<B>>, FifInfo)> {
        load_from_fif(fif_path, &self.data_cfg, data_norm, &self.device)
    }

    /// Encode a list of [`InputBatch`]es produced by [`Self::preprocess_fif`].
    pub fn encode_batches(
        &self,
        batches: Vec<InputBatch<B>>,
    ) -> anyhow::Result<Vec<EpochEmbedding>> {
        self.encode_inputs(batches)
    }

    /// Reference to the Burn device this encoder was loaded on.
    pub fn device(&self) -> &B::Device { &self.device }

    // ── Internal ──────────────────────────────────────────────────────────────

    pub(crate) fn encode_inputs(
        &self,
        batches: Vec<InputBatch<B>>,
    ) -> anyhow::Result<Vec<EpochEmbedding>> {
        batches.into_iter().map(|b| self.encode_one(b)).collect()
    }

    fn encode_one(&self, batch: InputBatch<B>) -> anyhow::Result<EpochEmbedding> {
        let n_channels = batch.n_channels;
        let tc         = batch.tc;

        // Keep a copy of tok_idx and chan_pos before the encoder consumes them.
        let tok_idx_saved  = batch.tok_idx.clone();
        let chan_pos_saved = batch.chan_pos.clone();

        // Run encoder: [1, S, output_dim]
        let enc_out = self.encoder.forward(
            batch.encoder_input,
            batch.tok_idx,
            &self.rope,
        );
        let [_, s, output_dim] = enc_out.dims();

        // Squeeze batch dim → [S, output_dim] and extract as Vec<f32>.
        let embeddings = enc_out
            .squeeze::<2>()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("embedding→vec: {e:?}"))?;

        // tok_idx [S, 4] → Vec<i64>.
        // NdArray backend stores Int as i64; wgpu backend stores Int as i32.
        // Try i64 first, fall back to i32 and widen.
        let tok_idx_data = tok_idx_saved.into_data();
        let tok_idx: Vec<i64> = tok_idx_data
            .to_vec::<i64>()
            .or_else(|_| tok_idx_data.to_vec::<i32>()
                .map(|v| v.into_iter().map(|x| x as i64).collect()))
            .map_err(|e| anyhow::anyhow!("tok_idx→vec: {e:?}"))?;

        // chan_pos [C, 3] → Vec<f32>.
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
    }
}
