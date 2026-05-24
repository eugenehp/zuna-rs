//! RLX-backed [`ZunaDecoder`] — same role as `crate::decoder::ZunaDecoder`
//! but built on `rlx::Session` / `rlx::CompiledGraph` instead of Burn.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use crate::config::ModelConfig;
use super::data as rdata;
use super::encoder::EpochEmbedding;
use super::graph::{build_decoder_graph, DecoderSpec};
use super::weights::{apply_params, build_decoder_params, load_safetensors, ParamMap};

/// One reconstructed epoch, mirroring `crate::inference::EpochOutput`.
#[derive(Clone, Debug)]
pub struct EpochOutput {
    pub reconstructed: Vec<f32>,
    pub shape:         Vec<usize>,
    pub chan_pos:      Vec<f32>,
    pub n_channels:    usize,
}

pub struct ZunaDecoder {
    pub model_cfg: ModelConfig,
    pub n_heads:   usize,
    pub device:    rlx::Device,
    pub global_sigma: f32,

    params:     ParamMap,
    rope_table: Vec<f32>,

    session: rlx::Session,
    cache:   HashMap<usize, rlx::CompiledGraph>,
}

impl ZunaDecoder {
    pub fn load(
        config_path:  &Path,
        weights_path: &Path,
        device:       rlx::Device,
    ) -> anyhow::Result<(Self, f64)> {
        let cfg_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading config: {}", config_path.display()))?;
        let hf_val: serde_json::Value = serde_json::from_str(&cfg_str)?;
        let model_cfg: ModelConfig = serde_json::from_value(hf_val["model"].clone())
            .context("parsing model config")?;

        let t = std::time::Instant::now();
        let mut raw = load_safetensors(
            weights_path.to_str().context("weights path not valid UTF-8")?,
        )?;
        let (params, n_heads) = build_decoder_params(&mut raw, &model_cfg)?;

        let rope_table = rdata::build_rope_table(
            model_cfg.head_dim, model_cfg.rope_dim,
            model_cfg.max_seqlen, model_cfg.rope_theta,
        );

        let global_sigma = model_cfg.stft_global_sigma as f32;
        let session = rlx::Session::new(device);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self {
            model_cfg, n_heads, device, global_sigma,
            params, rope_table, session, cache: HashMap::new(),
        }, ms))
    }

    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "ZUNA decoder (RLX, dev={:?})  dim={}  layers={}  head_dim={}  t_dim={}  σ={}",
            self.device, c.dim, c.n_layers, c.head_dim, c.t_dim, self.global_sigma,
        )
    }

    fn spec(&self, b: usize, s: usize) -> DecoderSpec {
        DecoderSpec {
            b, s,
            input_dim:   self.model_cfg.input_dim,
            encoder_dim: self.model_cfg.encoder_output_dim,
            dim:         self.model_cfg.dim,
            t_dim:       self.model_cfg.t_dim,
            n_layers:    self.model_cfg.n_layers,
            head_dim:    self.model_cfg.head_dim,
            n_heads:     self.n_heads,
            hidden_dim:  self.model_cfg.ffn_hidden_dim(),
            norm_eps:    self.model_cfg.norm_eps as f32,
        }
    }

    fn compiled_for(&mut self, b: usize, s: usize) -> &mut rlx::CompiledGraph {
        let key = b * 0x10_0000 + s;
        if !self.cache.contains_key(&key) {
            let spec = self.spec(b, s);
            let graph = build_decoder_graph(&spec);
            let mut compiled = self.session.compile(graph);
            apply_params(&mut compiled, &self.params);
            self.cache.insert(key, compiled);
        }
        self.cache.get_mut(&key).expect("just inserted")
    }

    /// Run one velocity-prediction step. All buffers are CPU-side `Vec<f32>`.
    pub fn forward_step(
        &mut self,
        z:        &[f32],   // [b, s, input_dim]
        enc_out:  &[f32],   // [b, s, encoder_output_dim]
        time_t:   f32,
        cos:      &[f32],   // [1, s, 1, head_dim/2]
        sin:      &[f32],
        b: usize, s: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let time_t_buf = vec![time_t; b];
        let compiled = self.compiled_for(b, s);
        let outs = compiled.run(&[
            ("z",         z),
            ("enc_out",   enc_out),
            ("time_t",    &time_t_buf),
            ("freqs_cos", cos),
            ("freqs_sin", sin),
        ]);
        outs.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("decoder graph produced no output"))
    }

    /// Reverse-time Euler diffusion loop, mirroring `EncoderDecoder::sample`.
    /// `noise_seed` lets the parity test fix the initial noise.
    pub fn decode_from_embedding(
        &mut self,
        ep:           &EpochEmbedding,
        steps:        usize,
        cfg:          f32,
        data_norm:    f32,
        noise_seed:   u64,
    ) -> anyhow::Result<EpochOutput> {
        let b  = 1usize;
        let s  = ep.shape[0];
        let id = self.model_cfg.input_dim;
        let ed = self.model_cfg.encoder_output_dim;
        let head_dim = self.model_cfg.head_dim;
        let rope_dim = self.model_cfg.rope_dim;
        anyhow::ensure!(ep.shape[1] == ed,
            "embedding output_dim mismatch: expected {ed}, got {}", ep.shape[1]);

        // RoPE freqs from the un-interleaved tok_idx (decoder operates on
        // a non-interleaved sequence).
        let (cos, sin) = rdata::precompute_rope(
            &ep.tok_idx, &self.rope_table, head_dim, rope_dim, s,
        );

        // Initial noise z ~ N(0, σ²).
        let sigma = self.global_sigma;
        let mut z = rdata::sample_normal(b * s * id, sigma, noise_seed);

        let dt = 1.0_f32 / steps as f32;
        for i in (1..=steps).rev() {
            let t_val = dt * i as f32;
            let vc = self.forward_step(&z, &ep.embeddings, t_val, &cos, &sin, b, s)?;
            let vc = if (cfg - 1.0).abs() > 1e-4 {
                let enc_zero = vec![0f32; b * s * ed];
                let vc_u = self.forward_step(&z, &enc_zero, t_val, &cos, &sin, b, s)?;
                vc_u.iter().zip(vc.iter())
                    .map(|(u, c)| u + cfg * (c - u))
                    .collect()
            } else {
                vc
            };
            for (zi, vi) in z.iter_mut().zip(vc.iter()) {
                *zi -= dt * vi;
            }
        }

        // Invert chop-and-reshape: [s, tf] is already row-major; just
        // re-stride into [n_channels, tc * tf].
        let tf = self.model_cfg.input_dim;
        let mut recon = vec![0f32; ep.n_channels * ep.tc * tf];
        for ch in 0..ep.n_channels {
            for tt in 0..ep.tc {
                let token = ch * ep.tc + tt;
                for f in 0..tf {
                    recon[ch * (ep.tc * tf) + tt * tf + f] = z[token * tf + f] * data_norm;
                }
            }
        }

        Ok(EpochOutput {
            reconstructed: recon,
            shape: vec![ep.n_channels, ep.tc * tf],
            chan_pos: ep.chan_pos.clone(),
            n_channels: ep.n_channels,
        })
    }
}
