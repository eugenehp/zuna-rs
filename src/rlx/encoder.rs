//! RLX-backed [`ZunaEncoder`] — same role as `crate::encoder::ZunaEncoder`
//! but built on `rlx::Session` / `rlx::CompiledGraph` instead of Burn.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;

use crate::config::ModelConfig;
use super::data as rdata;
use super::graph::{build_encoder_graph, EncoderSpec};
use super::weights::{apply_params, build_encoder_params, load_safetensors, ParamMap};

/// One encoder-side embedding, mirroring `crate::encoder::EpochEmbedding`.
#[derive(Clone, Debug)]
pub struct EpochEmbedding {
    pub embeddings: Vec<f32>,
    pub shape:      Vec<usize>,   // [s, output_dim]
    pub tok_idx:    Vec<i32>,     // [s, 4]
    pub chan_pos:   Vec<f32>,     // [n_channels, 3]
    pub n_channels: usize,
    pub tc:         usize,
}

/// RLX-backend encoder. Holds the session, parameter map, RoPE table,
/// and a per-shape compiled-graph cache.
pub struct ZunaEncoder {
    pub model_cfg: ModelConfig,
    pub n_heads:   usize,
    pub device:    rlx::Device,

    /// Parameter map (encoder side only) — graph-ready.
    params: ParamMap,
    /// `registers` weight kept separately so the CPU-side interleaver
    /// can copy it into the input buffer.
    registers: Vec<f32>,
    /// Shared RoPE table: `[max_seqlen, half, 4]` row-major.
    rope_table: Vec<f32>,

    session: rlx::Session,
    /// Cache of compiled graphs keyed by `s2 = s * (1 + downsample_factor)`.
    cache: HashMap<usize, rlx::CompiledGraph>,
}

impl ZunaEncoder {
    /// Load encoder weights from a HuggingFace `config.json` and
    /// `model.safetensors` for the given device. Returns `(self, weight_load_ms)`.
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
        let (mut params, n_heads) = build_encoder_params(&mut raw, &model_cfg)?;

        // Pull `encoder.registers` out of the param map: it isn't fed into
        // the graph (the interleaver consumes it on the CPU side).
        let registers = params.remove("encoder.registers")
            .map(|p| p.data)
            .unwrap_or_default();

        let rope_table = rdata::build_rope_table(
            model_cfg.head_dim, model_cfg.rope_dim,
            model_cfg.max_seqlen, model_cfg.rope_theta,
        );

        let session = rlx::Session::new(device);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((Self {
            model_cfg, n_heads, device,
            params, registers, rope_table,
            session, cache: HashMap::new(),
        }, ms))
    }

    /// One-line description (mirrors the Burn impl).
    pub fn describe(&self) -> String {
        let c = &self.model_cfg;
        format!(
            "ZUNA encoder (RLX, dev={:?})  dim={}  layers={}  head_dim={}  out_dim={}",
            self.device, c.dim, c.n_layers, c.head_dim, c.encoder_output_dim,
        )
    }

    /// Encoder spec for a given `(b, s)`.
    fn spec(&self, b: usize, s: usize) -> EncoderSpec {
        let df  = self.model_cfg.encoder_latent_downsample_factor;
        let s2  = s * (df + 1);
        EncoderSpec {
            b, s, s2,
            input_dim:  self.model_cfg.input_dim,
            output_dim: self.model_cfg.encoder_output_dim,
            dim:        self.model_cfg.dim,
            n_layers:   self.model_cfg.n_layers,
            head_dim:   self.model_cfg.head_dim,
            n_heads:    self.n_heads,
            hidden_dim: self.model_cfg.ffn_hidden_dim(),
            downsample_factor: df,
            norm_eps:   self.model_cfg.norm_eps as f32,
        }
    }

    fn compiled_for(&mut self, b: usize, s: usize) -> &mut rlx::CompiledGraph {
        let df = self.model_cfg.encoder_latent_downsample_factor;
        let s2 = s * (df + 1);
        let key = b * 0x10_0000 + s2;
        if !self.cache.contains_key(&key) {
            let spec = self.spec(b, s);
            let graph = build_encoder_graph(&spec);
            let mut compiled = self.session.compile(graph);
            apply_params(&mut compiled, &self.params);
            self.cache.insert(key, compiled);
        }
        self.cache.get_mut(&key).expect("just inserted")
    }

    /// Encode one preprocessed epoch: `token_values [b,s,input_dim] +
    /// tok_idx [s,4] + chan_pos [n_channels,3]` → `EpochEmbedding`.
    pub fn encode_one(
        &mut self,
        token_values: &[f32],
        tok_idx:      &[i32],
        chan_pos:     &[f32],
        n_channels:   usize,
        tc:           usize,
    ) -> anyhow::Result<EpochEmbedding> {
        let b  = 1usize;
        let s  = tok_idx.len() / 4;
        let df = self.model_cfg.encoder_latent_downsample_factor;
        let s2 = s * (df + 1);
        let head_dim = self.model_cfg.head_dim;
        let rope_dim = self.model_cfg.rope_dim;

        // CPU prep: interleave tokens with registers; build per-position
        // RoPE freqs from the repeated token index.
        let x = rdata::preinterleave(
            token_values, &self.registers, b, s, self.model_cfg.input_dim, df,
        );
        let tok_idx_x = rdata::repeat_token_idx(tok_idx, s, df);
        let (cos, sin) = rdata::precompute_rope(
            &tok_idx_x, &self.rope_table, head_dim, rope_dim, s2,
        );

        let cfg = self.model_cfg.clone();
        let out_dim = cfg.encoder_output_dim;
        let compiled = self.compiled_for(b, s);
        let outs = compiled.run(&[
            ("x",         &x),
            ("freqs_cos", &cos),
            ("freqs_sin", &sin),
        ]);
        let embeddings = outs.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("encoder graph produced no output"))?;

        Ok(EpochEmbedding {
            embeddings,
            shape: vec![s, out_dim],
            tok_idx: tok_idx.to_vec(),
            chan_pos: chan_pos.to_vec(),
            n_channels,
            tc,
        })
    }
}
