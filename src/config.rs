/// Model and runtime configuration for ZUNA inference.
///
/// `ModelConfig` is deserialised from the HuggingFace `config.json`
/// (the `"model"` sub-object).  Field names must match exactly.

// ── ModelConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelConfig {
    // Core transformer
    pub dim:      usize,    // 1024
    pub n_layers: usize,    // 16
    pub head_dim: usize,    // 64

    // Token I/O
    pub input_dim:          usize,  // 32
    pub encoder_output_dim: usize,  // 32

    // Encoder register/downsampling
    pub encoder_latent_downsample_factor: usize,  // 1

    // Decoder timestep conditioner output dim
    #[serde(default = "default_t_dim")]
    pub t_dim: usize,  // 64

    // Rotary embeddings
    pub max_seqlen:  usize,  // 50
    pub rope_dim:    usize,  // 4
    pub rope_theta:  f64,    // 10_000.0

    // Normalisation
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,  // 1e-5

    // Feed-forward rounding
    #[serde(default)]
    pub ffn_dim_multiplier: Option<f64>,
    #[serde(default = "default_multiple_of")]
    pub multiple_of: usize,  // 256

    // Diffusion noise std
    pub stft_global_sigma: f64,  // 0.1
}

fn default_t_dim()       -> usize { 64 }
fn default_norm_eps()    -> f64   { 1e-5 }
fn default_multiple_of() -> usize { 256 }

impl ModelConfig {
    /// n_heads is NOT dim/head_dim for this checkpoint.
    /// It must be inferred from the wq weight shape at load time.
    /// Use `WeightMap::infer_n_heads()` instead of calling this.
    pub fn n_heads_fallback(&self) -> usize { self.dim / self.head_dim }

    /// Feed-forward hidden dim (matches Python FeedForward.__init__):
    ///   hidden = int(2 * 4 * dim / 3)  →  2730
    ///   hidden = 256 * ceil(2730 / 256) →  2816
    pub fn ffn_hidden_dim(&self) -> usize {
        let mut h = (2 * 4 * self.dim) / 3;
        if let Some(m) = self.ffn_dim_multiplier {
            h = (m * h as f64) as usize;
        }
        self.multiple_of * ((h + self.multiple_of - 1) / self.multiple_of)
    }
}

// ── InferConfig ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InferConfig {
    pub sample_steps: usize,  // 50
    pub cfg:          f32,    // 1.0 (no guidance)
    pub data_norm:    f32,    // 10.0
}

impl Default for InferConfig {
    fn default() -> Self {
        Self { sample_steps: 50, cfg: 1.0, data_norm: 10.0 }
    }
}

// ── DataConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Fine time points per EEG token (= input_dim of the model).
    pub num_fine_time_pts: usize,  // 32
    /// Number of bins for x/y/z channel-position discretisation.
    pub num_bins: usize,           // 50
    /// Bounding box for scalp positions (metres), used in discretisation.
    pub xyz_min: [f32; 3],         // [-0.12, -0.12, -0.12]
    pub xyz_max: [f32; 3],         // [ 0.12,  0.12,  0.12]
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            num_fine_time_pts: 32,
            num_bins: 50,
            xyz_min: [-0.12, -0.12, -0.12],
            xyz_max: [ 0.12,  0.12,  0.12],
        }
    }
}
