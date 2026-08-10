//! Model and runtime configuration for ZUNA inference.
//!
//! `ModelConfig` is deserialised from the HuggingFace `config.json`
//! (the `"model"` sub-object).  Field names must match exactly.

// ── ModelConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    // Core transformer
    pub dim: usize,      // 1024
    pub n_layers: usize, // 16
    pub head_dim: usize, // 64

    // Token I/O
    pub input_dim: usize,          // 32
    pub encoder_output_dim: usize, // 32

    // Encoder register/downsampling
    pub encoder_latent_downsample_factor: usize, // 1

    // Decoder timestep conditioner output dim
    #[serde(default = "default_t_dim")]
    pub t_dim: usize, // 64

    // Rotary embeddings
    pub max_seqlen: usize, // 50
    pub rope_dim: usize,   // 4
    pub rope_theta: f64,   // 10_000.0

    // Normalisation
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64, // 1e-5

    // Feed-forward rounding
    #[serde(default)]
    pub ffn_dim_multiplier: Option<f64>,
    #[serde(default = "default_multiple_of")]
    pub multiple_of: usize, // 256

    // Diffusion noise std
    pub stft_global_sigma: f64, // 0.1

    // ── Settings we don't implement; parsed only so `validate` can reject them ──
    /// Absolute positional embedding width. ZUNA1/ZUNA1.1 ship `0` (no APE).
    #[serde(default)]
    pub ape_dim: usize,
    /// Token-index layout fed to the axial RoPE.
    #[serde(default = "default_tok_idx_type")]
    pub tok_idx_type: String,
    /// When true the decoder's cross-attention key norm is a plain RMSNorm
    /// rather than an AdaRMSNorm. ZUNA1/ZUNA1.1 ship `false`.
    #[serde(default)]
    pub seqlen_t: bool,
    /// When true `x,y,z` are zeroed in `tok_idx`. ZUNA1/ZUNA1.1 ship `false`.
    #[serde(default)]
    pub zero_spatial: bool,
}

fn default_t_dim() -> usize {
    64
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_multiple_of() -> usize {
    256
}
fn default_tok_idx_type() -> String {
    "{x,y,z,tc}".to_string()
}

impl ModelConfig {
    /// Reject checkpoints whose `config.json` enables a feature this crate
    /// does not implement, instead of silently producing wrong outputs.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.ape_dim == 0,
            "ape_dim={} — absolute position embeddings are not implemented",
            self.ape_dim
        );
        anyhow::ensure!(
            self.tok_idx_type == "{x,y,z,tc}",
            "tok_idx_type={:?} — only \"{{x,y,z,tc}}\" is implemented",
            self.tok_idx_type
        );
        anyhow::ensure!(
            !self.seqlen_t,
            "seqlen_t=true — plain-RMSNorm cross-attention key norm is not implemented"
        );
        anyhow::ensure!(!self.zero_spatial, "zero_spatial=true is not implemented");
        anyhow::ensure!(
            self.rope_dim == 4,
            "rope_dim={} — only 4-D axial RoPE is implemented",
            self.rope_dim
        );
        Ok(())
    }

    /// n_heads is NOT dim/head_dim for this checkpoint.
    /// It must be inferred from the wq weight shape at load time.
    /// Use `WeightMap::infer_n_heads()` instead of calling this.
    pub fn n_heads_fallback(&self) -> usize {
        self.dim / self.head_dim
    }

    /// Feed-forward hidden dim (matches Python FeedForward.__init__):
    ///   hidden = int(2 * 4 * dim / 3)  →  2730
    ///   hidden = 256 * ceil(2730 / 256) →  2816
    pub fn ffn_hidden_dim(&self) -> usize {
        let mut h = (2 * 4 * self.dim) / 3;
        if let Some(m) = self.ffn_dim_multiplier {
            h = (m * h as f64) as usize;
        }
        self.multiple_of * h.div_ceil(self.multiple_of)
    }
}

// ── ModelArch ─────────────────────────────────────────────────────────────────

/// Architecture toggles that differ between ZUNA checkpoints.
///
/// Neither flag appears in `config.json` — upstream hard-codes both in
/// `lingua/transformer.py` (`do_QK_norm`, `do_sandwich_norm`) — so they are
/// detected from the tensor names in the safetensors file instead.
///
/// | checkpoint | `qk_norm` | `sandwich_norm` |
/// |---|---|---|
/// | [`Zyphra/ZUNA`](https://huggingface.co/Zyphra/ZUNA)         | no  | no  |
/// | [`Zyphra/ZUNA1.1`](https://huggingface.co/Zyphra/ZUNA1.1)   | yes | yes |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelArch {
    /// Per-head RMSNorm (`head_dim`-wide) applied to Q and K **before** RoPE.
    /// Weights: `<attn>.q_norm.weight` / `<attn>.k_norm.weight`.
    pub qk_norm: bool,
    /// "Sandwich" RMSNorm (`dim`-wide) applied to each sub-layer output
    /// **before** the residual add. Weights: `<sublayer>_norm_post.weight`.
    pub sandwich_norm: bool,
}

impl ModelArch {
    /// `Zyphra/ZUNA` — pre-norm only.
    pub const ZUNA1: Self = Self {
        qk_norm: false,
        sandwich_norm: false,
    };
    /// `Zyphra/ZUNA1.1` — QK-norm plus sandwich norm on every sub-layer.
    pub const ZUNA1_1: Self = Self {
        qk_norm: true,
        sandwich_norm: true,
    };

    /// Detect the architecture from the checkpoint's tensor names.
    ///
    /// `has_key` is queried with **canonical** keys — i.e. after
    /// [`canonical_key`] has collapsed the extra `norm.` segment that
    /// ZUNA1.1's `torch.nn.RMSNorm` wrapper introduces.
    pub fn detect<F: Fn(&str) -> bool>(has_key: F) -> Self {
        Self {
            qk_norm: has_key("encoder.layers.0.attention.q_norm.weight"),
            sandwich_norm: has_key("encoder.layers.0.attention_norm_post.weight"),
        }
    }

    /// Human-readable name for logs.
    pub fn label(&self) -> &'static str {
        match *self {
            Self::ZUNA1 => "ZUNA1",
            Self::ZUNA1_1 => "ZUNA1.1",
            _ => "ZUNA (custom)",
        }
    }
}

/// Normalise a safetensors key so ZUNA1 and ZUNA1.1 share one namespace.
///
/// ZUNA1 stored plain RMSNorm scales directly on the module
/// (`…attention_norm.weight`); ZUNA1.1 wraps `torch.nn.RMSNorm` in a
/// `self.norm` sub-module, so the same scale lands at
/// `…attention_norm.norm.weight`. Collapsing the duplicated `norm.` keeps
/// every downstream lookup (graph param names, Burn loaders) version-agnostic.
///
/// A `.norm.weight` suffix is only collapsed when the module it hangs off is
/// itself a norm (`attention_norm`, `ffn_norm_post`, `q_norm`, …). That leaves
/// ZUNA1's `encoder.norm.weight` — where `norm` *is* the module — and the
/// AdaRMSNorm linears (`….weight.weight`) untouched.
pub fn canonical_key(key: &str) -> std::borrow::Cow<'_, str> {
    let Some(head) = key.strip_suffix(".norm.weight") else {
        return std::borrow::Cow::Borrowed(key);
    };
    let module = head.rsplit('.').next().unwrap_or(head);
    if module.contains("norm") {
        std::borrow::Cow::Owned(format!("{head}.weight"))
    } else {
        std::borrow::Cow::Borrowed(key)
    }
}

// ── InferConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InferConfig {
    pub sample_steps: usize, // 50
    pub cfg: f32,            // 1.0 (no guidance)
    pub data_norm: f32,      // 10.0
}

impl Default for InferConfig {
    fn default() -> Self {
        Self {
            sample_steps: 50,
            cfg: 1.0,
            data_norm: 10.0,
        }
    }
}

// ── DataConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Fine time points per EEG token (= input_dim of the model).
    pub num_fine_time_pts: usize, // 32
    /// Number of bins for x/y/z channel-position discretisation.
    pub num_bins: usize, // 50
    /// Bounding box for scalp positions (metres), used in discretisation.
    pub xyz_min: [f32; 3], // [-0.12, -0.12, -0.12]
    pub xyz_max: [f32; 3], // [ 0.12,  0.12,  0.12]
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            num_fine_time_pts: 32,
            num_bins: 50,
            xyz_min: [-0.12, -0.12, -0.12],
            xyz_max: [0.12, 0.12, 0.12],
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The `norm.` collapse must fire on ZUNA1.1's doubled spelling and on
    /// nothing else — in particular not on ZUNA1's `encoder.norm.weight`,
    /// whose `norm` is the module name, nor on the AdaRMSNorm linears.
    #[test]
    fn canonical_key_collapses_only_the_doubled_norm() {
        let cases = [
            // ZUNA1.1 → ZUNA1 spelling
            (
                "encoder.layers.0.attention_norm.norm.weight",
                "encoder.layers.0.attention_norm.weight",
            ),
            (
                "encoder.layers.0.ffn_norm.norm.weight",
                "encoder.layers.0.ffn_norm.weight",
            ),
            ("encoder.norm.norm.weight", "encoder.norm.weight"),
            (
                "encoder.layers.0.attention.q_norm.norm.weight",
                "encoder.layers.0.attention.q_norm.weight",
            ),
            (
                "decoder.layers.0.ffn_norm_post.norm.weight",
                "decoder.layers.0.ffn_norm_post.weight",
            ),
            // Already canonical / unrelated — must pass through untouched
            ("encoder.norm.weight", "encoder.norm.weight"),
            (
                "encoder.layers.0.attention_norm.weight",
                "encoder.layers.0.attention_norm.weight",
            ),
            ("decoder.norm.weight.weight", "decoder.norm.weight.weight"),
            ("decoder.norm.weight.bias", "decoder.norm.weight.bias"),
            (
                "decoder.layers.0.attention_norm.weight.bias",
                "decoder.layers.0.attention_norm.weight.bias",
            ),
            (
                "encoder.tok_embeddings.weight",
                "encoder.tok_embeddings.weight",
            ),
            ("encoder.registers", "encoder.registers"),
        ];
        for (input, want) in cases {
            assert_eq!(
                canonical_key(input).as_ref(),
                want,
                "canonical_key({input:?})"
            );
        }
    }

    #[test]
    fn arch_detection_distinguishes_the_checkpoints() {
        let zuna1: &[&str] = &[
            "encoder.layers.0.attention_norm.weight",
            "encoder.layers.0.attention.wq.weight",
        ];
        let zuna11: &[&str] = &[
            "encoder.layers.0.attention_norm.weight",
            "encoder.layers.0.attention.wq.weight",
            "encoder.layers.0.attention.q_norm.weight",
            "encoder.layers.0.attention_norm_post.weight",
        ];
        assert_eq!(ModelArch::detect(|k| zuna1.contains(&k)), ModelArch::ZUNA1);
        assert_eq!(
            ModelArch::detect(|k| zuna11.contains(&k)),
            ModelArch::ZUNA1_1
        );
        assert_eq!(ModelArch::ZUNA1.label(), "ZUNA1");
        assert_eq!(ModelArch::ZUNA1_1.label(), "ZUNA1.1");
    }

    fn parse(json: &str) -> ModelConfig {
        serde_json::from_str(json).expect("config parses")
    }

    /// Both shipped `config.json` files must parse and validate; the fields
    /// ZUNA1.1 added (and ZUNA1 omits) fall back to their defaults.
    #[test]
    fn both_checkpoint_configs_parse() {
        let zuna1 = parse(
            r#"{
            "dim": 1024, "n_layers": 16, "head_dim": 64,
            "input_dim": 32, "encoder_input_dim": 32, "encoder_output_dim": 32,
            "encoder_latent_downsample_factor": 1,
            "max_seqlen": 50, "rope_dim": 4, "rope_theta": 10000.0,
            "tok_idx_type": "{x,y,z,tc}", "stft_global_sigma": 0.1,
            "adaptive_loss_weighting": true, "dropout_type": "zeros"
        }"#,
        );
        assert_eq!(zuna1.max_seqlen, 50);
        assert_eq!(zuna1.ape_dim, 0);
        assert_eq!(zuna1.t_dim, 64);
        assert_eq!(zuna1.ffn_hidden_dim(), 2816);
        zuna1.validate().expect("ZUNA1 config is supported");

        let zuna11 = parse(
            r#"{
            "dim": 1024, "n_layers": 16, "head_dim": 64, "seqlen_t": false,
            "input_dim": 32, "encoder_input_dim": 32, "encoder_output_dim": 32,
            "encoder_latent_downsample_factor": 1,
            "max_seqlen": 256, "max_chans": 512, "model_dtype": "bf16",
            "rope_dim": 4, "rope_theta": 10000.0,
            "ape_dim": 0, "ape_theta": 10000.0,
            "ape_embedding_type": "scaled_dim_sinusoidal",
            "tok_idx_type": "{x,y,z,tc}", "zero_spatial": false,
            "stft_global_sigma": 0.1, "num_fine_time_pts": 32,
            "dropout_vec_type": "zeros", "register_tok_idx": "mean_all"
        }"#,
        );
        assert_eq!(zuna11.max_seqlen, 256);
        assert_eq!(zuna11.ffn_hidden_dim(), 2816);
        zuna11.validate().expect("ZUNA1.1 config is supported");
    }

    #[test]
    fn validate_rejects_unimplemented_settings() {
        let cfg = |extra: &str| {
            parse(&format!(
                r#"{{
            "dim": 1024, "n_layers": 16, "head_dim": 64,
            "input_dim": 32, "encoder_output_dim": 32,
            "encoder_latent_downsample_factor": 1,
            "max_seqlen": 256, "rope_theta": 10000.0,
            "stft_global_sigma": 0.1, {extra}
        }}"#
            ))
        };
        assert!(cfg(r#""rope_dim": 4"#).validate().is_ok());
        for extra in [
            r#""rope_dim": 4, "ape_dim": 64"#,
            r#""rope_dim": 4, "seqlen_t": true"#,
            r#""rope_dim": 4, "zero_spatial": true"#,
            r#""rope_dim": 4, "tok_idx_type": "{x,y,z,tc,ch}""#,
            r#""rope_dim": 1"#,
        ] {
            assert!(cfg(extra).validate().is_err(), "should reject {extra}");
        }
    }
}
