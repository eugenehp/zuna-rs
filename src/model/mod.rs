pub mod attention;
pub mod conditioner;
pub mod cross_attention;
pub mod decoder;
pub mod decoder_block;
pub mod encoder;
pub mod encoder_block;
pub mod encoder_decoder;
pub mod feedforward;
pub mod norm;
pub mod rope;

use burn::module::{Param, ParamId};
use burn::nn::Linear;
use burn::prelude::*;

use crate::config::{ModelArch, ModelConfig};

// ── Geometry ──────────────────────────────────────────────────────────────────
//
// The module constructors take these instead of a dozen loose `usize`s, so a
// transposed pair of arguments is a compile error rather than a silent shape
// bug. All three are `Copy`, so passing them costs nothing.

/// Geometry of one transformer block, shared by the encoder and decoder stacks.
#[derive(Clone, Copy, Debug)]
pub struct BlockDims {
    /// Residual-stream width (1024).
    pub dim: usize,
    /// Per-head width (64).
    pub head_dim: usize,
    /// Query heads — inferred from the `wq` weight shape, not `config.json`.
    pub n_heads: usize,
    /// Key/value heads. ZUNA uses full MHA, so this equals `n_heads`.
    pub n_kv_heads: usize,
    /// SwiGLU hidden width (2816).
    pub hidden_dim: usize,
    pub norm_eps: f64,
    /// Which extra norms this checkpoint carries.
    pub arch: ModelArch,
}

/// Geometry of the full encoder stack.
#[derive(Clone, Copy, Debug)]
pub struct EncoderDims {
    pub block: BlockDims,
    /// Fine time points per token (32).
    pub input_dim: usize,
    /// Latent width per register token (32).
    pub output_dim: usize,
    pub n_layers: usize,
    /// Real tokens per register (1).
    pub downsample_factor: usize,
}

/// Geometry of the full decoder stack.
#[derive(Clone, Copy, Debug)]
pub struct DecoderDims {
    pub block: BlockDims,
    pub input_dim: usize,
    /// Width of the encoder latent this decoder cross-attends to.
    pub encoder_dim: usize,
    /// Timestep-conditioner width (64).
    pub t_dim: usize,
    pub n_layers: usize,
}

impl BlockDims {
    /// Derive block geometry from a parsed `config.json`.
    ///
    /// `n_heads` and `arch` come from the weight file — see
    /// [`WeightMap::infer_n_heads`](crate::weights::WeightMap::infer_n_heads)
    /// and [`ModelArch::detect`].
    pub fn from_config(cfg: &ModelConfig, n_heads: usize, arch: ModelArch) -> Self {
        Self {
            dim: cfg.dim,
            head_dim: cfg.head_dim,
            n_heads,
            n_kv_heads: n_heads,
            hidden_dim: cfg.ffn_hidden_dim(),
            norm_eps: cfg.norm_eps,
            arch,
        }
    }
}

impl EncoderDims {
    /// Derive encoder geometry from a parsed `config.json`.
    pub fn from_config(cfg: &ModelConfig, n_heads: usize, arch: ModelArch) -> Self {
        Self {
            block: BlockDims::from_config(cfg, n_heads, arch),
            input_dim: cfg.input_dim,
            output_dim: cfg.encoder_output_dim,
            n_layers: cfg.n_layers,
            downsample_factor: cfg.encoder_latent_downsample_factor,
        }
    }
}

impl DecoderDims {
    /// Derive decoder geometry from a parsed `config.json`.
    pub fn from_config(cfg: &ModelConfig, n_heads: usize, arch: ModelArch) -> Self {
        Self {
            block: BlockDims::from_config(cfg, n_heads, arch),
            input_dim: cfg.input_dim,
            encoder_dim: cfg.encoder_output_dim,
            t_dim: cfg.t_dim,
            n_layers: cfg.n_layers,
        }
    }
}

/// Epsilon of the ZUNA1.1 QK-norms.
///
/// Upstream hard-codes `RMSNorm(head_dim, eps=1e-5)` in `Attention.__init__`
/// / `CrossAttention.__init__` rather than threading `args.norm_eps` through,
/// so this stays a constant even if `config.json` sets a different `norm_eps`.
pub const QK_NORM_EPS: f64 = 1e-5;

/// Create a [`Linear`] layer with **zero-filled** weights instead of the default
/// random (KaimingUniform) initialization.
///
/// This is used when weights will be immediately overwritten from a safetensors
/// file.  `Tensor::zeros` is essentially free compared to the ChaCha12-based
/// random fill that `LinearConfig::init` performs.
pub fn linear_zeros<B: Backend>(
    d_input: usize,
    d_output: usize,
    bias: bool,
    device: &B::Device,
) -> Linear<B> {
    let weight = Param::initialized(ParamId::new(), Tensor::zeros([d_input, d_output], device));
    let bias = if bias {
        Some(Param::initialized(
            ParamId::new(),
            Tensor::zeros([d_output], device),
        ))
    } else {
        None
    };
    Linear { weight, bias }
}
