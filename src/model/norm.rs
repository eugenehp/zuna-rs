/// RMSNorm and AdaRMSNorm (burn 0.20.1)
///
/// burn 0.20.1 ships `burn::nn::RmsNorm` natively (field: `gamma`).
/// We use it directly for the encoder, and implement AdaRMSNorm by hand
/// for the decoder (conditioned on timestep embedding `c`).

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};

// ── Plain RMSNorm wrapper (maps the API used in the rest of our code) ─────────

/// Thin wrapper around `burn::nn::RmsNorm` exposing the same `forward` signature
/// used throughout this crate.
#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub inner: RmsNorm<B>,
}

impl<B: Backend> RMSNorm<B> {
    pub fn new(dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            inner: RmsNormConfig::new(dim).with_epsilon(eps).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x)
    }
}

// ── Adaptive RMSNorm (conditioned on timestep `c`) ────────────────────────────

/// `AdaRMSNorm` from xattn.py.
///
///   class AdaRMSNorm(nn.Module):
///     def __init__(self, emb_dim, dim, eps=1e-6):
///       self.weight = nn.Linear(emb_dim, dim, bias=True)   # named "weight" in state dict!
///     def forward(self, x, c):
///       normed = x * rsqrt(mean(x²,-1,keepdim=True) + eps)
///       return normed * self.weight(c)
///
/// The inner Linear's state-dict key is `<path>.weight.weight` / `<path>.weight.bias`
/// because PyTorch names the sub-module "weight".
#[derive(Module, Debug)]
pub struct AdaRMSNorm<B: Backend> {
    /// `nn.Linear(emb_dim, dim, bias=True)` — named "weight" in PyTorch.
    pub weight: Linear<B>,
    pub eps: f64,
}

impl<B: Backend> AdaRMSNorm<B> {
    pub fn new(emb_dim: usize, dim: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: LinearConfig::new(emb_dim, dim).with_bias(true).init(device),
            eps,
        }
    }

    /// x: [1, S, dim],  c: [1, 1, t_dim]  →  [1, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 3>) -> Tensor<B, 3> {
        let eps = self.eps as f32;
        // RMS normalise along the last dimension.
        // burn's mean_dim is keepdim, so rms shape is [1, S, 1].
        let rms = (x.clone().powf_scalar(2.0f32).mean_dim(2) + eps).sqrt();
        let normed = x / rms;
        // Adaptive scale: Linear(c) → [1, 1, dim], broadcast over S
        normed * self.weight.forward(c)
    }
}
