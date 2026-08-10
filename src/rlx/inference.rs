//! End-to-end RLX-backed inference (encoder → diffusion decode).
//!
//! Mirrors the Burn-side `crate::inference::ZunaInference` but holds
//! independent RLX encoder and decoder wrappers (both reuse the same
//! safetensors file at load time).

use std::path::Path;

use anyhow::Context;

use super::decoder::{EpochOutput, ZunaDecoder};
use super::encoder::{EpochEmbedding, ZunaEncoder};
use crate::config::{DataConfig, ModelArch, ModelConfig};
use crate::data::PreprocessedEpoch;

pub struct InferenceResult {
    pub epochs: Vec<EpochOutput>,
    pub ms_encode: f64,
    pub ms_decode: f64,
}

pub struct ZunaInference {
    pub encoder: ZunaEncoder,
    pub decoder: ZunaDecoder,
    pub data_cfg: DataConfig,
}

impl ZunaInference {
    /// Load both halves of the model for the given device.
    pub fn load(
        config_path: &Path,
        weights_path: &Path,
        device: rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let t = std::time::Instant::now();
        let (encoder, _) = ZunaEncoder::load(config_path, weights_path, device)?;
        let (decoder, _) = ZunaDecoder::load(config_path, weights_path, device)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        Ok((
            Self {
                encoder,
                decoder,
                data_cfg: DataConfig::default(),
            },
            ms,
        ))
    }

    pub fn model_cfg(&self) -> &ModelConfig {
        &self.encoder.model_cfg
    }

    /// Which ZUNA checkpoint was loaded, detected from the weight names.
    pub fn arch(&self) -> ModelArch {
        self.encoder.arch
    }

    /// Encode + decode a single preprocessed epoch.
    pub fn run_epoch(
        &mut self,
        ep: &PreprocessedEpoch,
        steps: usize,
        cfg: f32,
        data_norm: f32,
        noise_seed: u64,
    ) -> anyhow::Result<EpochOutput> {
        let embedding = self
            .encoder
            .encode_one(
                &ep.eeg_tokens,
                &ep.tok_idx,
                &ep.chan_pos,
                ep.n_channels,
                ep.tc,
            )
            .context("encoder.encode_one")?;
        self.decoder
            .decode_from_embedding(&embedding, steps, cfg, data_norm, noise_seed)
    }

    /// Encode-only path. Useful for tests / parity that only validate
    /// the encoder.
    pub fn encode_epoch(&mut self, ep: &PreprocessedEpoch) -> anyhow::Result<EpochEmbedding> {
        self.encoder.encode_one(
            &ep.eeg_tokens,
            &ep.tok_idx,
            &ep.chan_pos,
            ep.n_channels,
            ep.tc,
        )
    }
}
