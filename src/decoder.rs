//! Standalone ZUNA decoder — reconstruct EEG signals from latent embeddings.
//!
//! Use [`ZunaDecoder`] when you want to run only the decoder half of the model,
//! for example to reconstruct signals from embeddings that were previously
//! computed with [`crate::encoder::ZunaEncoder`] and saved to disk.
//!
//! # Example
//!
//! ```rust,ignore
//! use zuna_rs::decoder::ZunaDecoder;
//! use zuna_rs::encoder::EncodingResult;
//!
//! // Load stored embeddings from a previous encode pass.
//! let embeddings = EncodingResult::load_safetensors("data/embeddings.safetensors")?;
//!
//! // Load decoder weights only.
//! let (dec, ms) = ZunaDecoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//!
//! // Decode: run the rectified-flow diffusion loop conditioned on embeddings.
//! let result = dec.decode_embeddings(&embeddings, 50, 1.0, 10.0)?;
//! result.save_safetensors("output.safetensors")?;
//! ```

use std::{path::Path, time::Instant};

use anyhow::Context;
use burn::{prelude::*, tensor::Distribution};

use crate::{
    config::{DataConfig, ModelConfig},
    data::invert_reshape,
    encoder::{EncodingResult, EpochEmbedding},
    inference::{EpochOutput, InferenceResult},
    model::{decoder::DecoderTransformer, rope::RotaryEmbedding},
    weights::load_decoder_weights,
};

// ── ZunaDecoder ───────────────────────────────────────────────────────────────

/// Standalone ZUNA decoder.
///
/// Reconstructs EEG signals from latent embeddings produced by
/// [`crate::encoder::ZunaEncoder`] using a rectified-flow diffusion loop.
///
/// Load with [`ZunaDecoder::load`] (decoder weights only — saves ~50 % memory
/// vs the full [`crate::ZunaInference`]).
pub struct ZunaDecoder<B: Backend> {
    decoder:       DecoderTransformer<B>,
    rope:          RotaryEmbedding<B>,
    /// Architecture hyperparameters (from config.json).
    pub model_cfg: ModelConfig,
    /// Preprocessing / tokenisation parameters.
    pub data_cfg:  DataConfig,
    /// Diffusion noise standard deviation (σ).
    pub global_sigma: f32,
    device:        B::Device,
}

impl<B: Backend> ZunaDecoder<B> {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Load decoder weights from a HuggingFace `config.json` and
    /// `model.safetensors`.  Encoder tensors are read from disk but not kept
    /// in memory.
    ///
    /// Returns `(decoder, weight_load_ms)`.
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
        let (decoder, n_heads) = load_decoder_weights::<B>(
            &model_cfg,
            weights_path.to_str().context("weights path not valid UTF-8")?,
            &device,
        )?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("Detected n_heads = {n_heads}");

        let global_sigma = model_cfg.stft_global_sigma as f32;

        Ok((Self { decoder, rope, model_cfg, data_cfg: DataConfig::default(), global_sigma, device }, ms))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// One-line description of the loaded decoder.
    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "ZUNA decoder  dim={}  layers={}  head_dim={}  t_dim={}  σ={}",
            c.dim, c.n_layers, c.head_dim, c.t_dim, self.global_sigma,
        )
    }

    // ── High-level decode API ─────────────────────────────────────────────────

    /// Reconstruct EEG signals from pre-computed embeddings.
    ///
    /// This is the pure decode path — no FIF loading or preprocessing happens
    /// here.  The [`EncodingResult`] must have been produced by
    /// [`ZunaEncoder::encode_fif`] or [`ZunaEncoder::encode_batch`].
    ///
    /// # Arguments
    /// - `embeddings`  — pre-computed encoder latents
    /// - `steps`       — diffusion denoising steps (50 = full quality, 10 = fast)
    /// - `cfg`         — classifier-free guidance scale (1.0 = disabled)
    /// - `data_norm`   — divisor used during preprocessing; multiplied back into
    ///                   the output to restore the original signal scale
    pub fn decode_embeddings(
        &self,
        embeddings: &EncodingResult,
        steps:      usize,
        cfg:        f32,
        data_norm:  f32,
    ) -> anyhow::Result<InferenceResult> {
        let t_dec = Instant::now();
        let epochs = embeddings.epochs
            .iter()
            .map(|ep| self.decode_one(ep, steps, cfg, data_norm))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ms_infer = t_dec.elapsed().as_secs_f64() * 1000.0;

        Ok(InferenceResult {
            epochs,
            fif_info:   None,
            ms_preproc: 0.0,
            ms_infer,
        })
    }

    /// Decode a single epoch from a raw encoder output tensor `[1, S, output_dim]`.
    ///
    /// `tok_idx` is `[S, 4]`.  Returns the reconstructed token matrix
    /// `[1, S, input_dim]` **before** inversion of the chop-and-reshape.
    pub fn decode_tensor(
        &self,
        enc_out:   Tensor<B, 3>,
        tok_idx:   Tensor<B, 2, Int>,
        steps:     usize,
        cfg:       f32,
    ) -> Tensor<B, 3> {
        let device = enc_out.device();
        let [b, s, d] = enc_out.dims();
        let dt = 1.0_f32 / steps as f32;

        // Initial noise z ~ N(0, σ²)
        let sigma = self.global_sigma as f64;
        let mut z = Tensor::<B, 3>::random(
            [b, s, d],
            Distribution::Normal(0.0, sigma),
            &device,
        );

        // Rectified-flow Euler sampling loop
        for i in (1..=steps).rev() {
            let t_val  = dt * i as f32;
            let time_t = Tensor::<B, 3>::full([b, 1, 1], t_val, &device);

            let vc = self.decoder.forward(
                z.clone(), enc_out.clone(), time_t.clone(), tok_idx.clone(), &self.rope,
            );

            let vc = if (cfg - 1.0).abs() > 1e-4 {
                // Classifier-free guidance: run unconditioned pass with zeros
                let enc_zeros  = Tensor::zeros([b, s, d], &device);
                let vc_uncond  = self.decoder.forward(
                    z.clone(), enc_zeros, time_t, tok_idx.clone(), &self.rope,
                );
                vc_uncond.clone() + (vc - vc_uncond).mul_scalar(cfg)
            } else {
                vc
            };

            z = z - vc.mul_scalar(dt);
        }

        z
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn decode_one(
        &self,
        ep:        &EpochEmbedding,
        steps:     usize,
        cfg:       f32,
        data_norm: f32,
    ) -> anyhow::Result<EpochOutput> {
        let n_tokens = ep.n_tokens();
        let dc       = &self.data_cfg;

        // Reconstruct enc_out [1, S, output_dim] from stored Vec<f32>.
        let enc_out = Tensor::<B, 2>::from_data(
            TensorData::new(ep.embeddings.clone(), ep.shape.clone()),
            &self.device,
        )
        .unsqueeze_dim::<3>(0);  // [1, S, output_dim]

        // Reconstruct tok_idx [S, 4] from stored Vec<i64>.
        let tok_idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ep.tok_idx.clone(), vec![n_tokens, 4]),
            &self.device,
        );

        // Run diffusion; returns [1, S, input_dim].
        let z = self.decode_tensor(enc_out, tok_idx, steps, cfg);

        // Invert chop-and-reshape: [1, S, tf] → [C, T].
        let [_, s, tf] = z.dims();
        let recon = invert_reshape(
            z.reshape([s, tf]),
            ep.n_channels,
            ep.tc,
            dc.num_fine_time_pts,
        );
        let recon = recon.mul_scalar(data_norm);

        let shape         = recon.dims().to_vec();
        let reconstructed = recon
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("recon→vec: {e:?}"))?;
        let chan_pos = ep.chan_pos.clone();

        Ok(EpochOutput { reconstructed, shape, chan_pos, n_channels: ep.n_channels })
    }
}
