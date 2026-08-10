//! Self-Attention with 4-D axial RoPE (burn 0.20.1)
//!
//! Python: `Attention` in lingua/transformer.py.
//! Single-sample path: full attention, no document mask needed.
use crate::model::norm::RMSNorm;
use crate::model::rope::apply_rope;
use crate::model::{linear_zeros, QK_NORM_EPS};
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    /// ZUNA1.1 QK-norm: per-head RMSNorm on Q, applied before RoPE.
    pub q_norm: Option<RMSNorm<B>>,
    /// ZUNA1.1 QK-norm: per-head RMSNorm on K, applied before RoPE.
    pub k_norm: Option<RMSNorm<B>>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        qk_norm: bool,
        device: &B::Device,
    ) -> Self {
        let z = |i, o| linear_zeros(i, o, false, device);
        let qk = || qk_norm.then(|| RMSNorm::new(head_dim, QK_NORM_EPS, device));
        Self {
            wq: z(dim, n_heads * head_dim),
            wk: z(dim, n_kv_heads * head_dim),
            wv: z(dim, n_kv_heads * head_dim),
            wo: z(n_heads * head_dim, dim),
            q_norm: qk(),
            k_norm: qk(),
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// x:       [B, S, dim]
    /// freqs_4d: [S, head_dim/2, 2, 2]  (broadcasts over B)
    /// Returns: [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>, freqs_4d: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);

        let xq = self.wq.forward(x.clone()).reshape([b, s, h, dh]);
        let xk = self.wk.forward(x.clone()).reshape([b, s, h, dh]);
        let xv = self.wv.forward(x).reshape([b, s, h, dh]);

        // QK-norm (ZUNA1.1) runs on the [B,S,H,Dh] view, before RoPE.
        let xq = match &self.q_norm {
            Some(n) => n.forward(xq),
            None => xq,
        };
        let xk = match &self.k_norm {
            Some(n) => n.forward(xk),
            None => xk,
        };

        let (xq, xk) = apply_rope(xq, xk, freqs_4d);

        // [1, H, S, Dh] for matmul
        let xq = xq.swap_dims(1, 2); // [1, H, S, Dh]
        let xk = xk.swap_dims(1, 2);
        let xv = xv.swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn = softmax(xq.matmul(xk.transpose()).mul_scalar(scale), 3);
        let out = attn.matmul(xv); // [1, H, S, Dh]

        self.wo.forward(out.swap_dims(1, 2).reshape([b, s, h * dh]))
    }
}
