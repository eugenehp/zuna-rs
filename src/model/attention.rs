/// Self-Attention with 4-D axial RoPE (burn 0.20.1)
///
/// Python: `Attention` in lingua/transformer.py.
/// Single-sample path: full attention, no document mask needed.
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use crate::model::rope::apply_rope;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    pub n_heads:    usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        device: &B::Device,
    ) -> Self {
        let nobias = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        Self {
            wq: nobias(dim, n_heads    * head_dim),
            wk: nobias(dim, n_kv_heads * head_dim),
            wv: nobias(dim, n_kv_heads * head_dim),
            wo: nobias(n_heads * head_dim, dim),
            n_heads, n_kv_heads, head_dim,
        }
    }

    /// x:       [1, S, dim]
    /// freqs_4d: [S, head_dim/2, 2, 2]
    /// Returns: [1, S, dim]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        freqs_4d: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let (h, dh) = (self.n_heads, self.head_dim);

        let xq = self.wq.forward(x.clone()).reshape([b, s, h, dh]);
        let xk = self.wk.forward(x.clone()).reshape([b, s, h, dh]);
        let xv = self.wv.forward(x).reshape([b, s, h, dh]);

        let (xq, xk) = apply_rope(xq, xk, freqs_4d);

        // [1, H, S, Dh] for matmul
        let xq = xq.swap_dims(1, 2);  // [1, H, S, Dh]
        let xk = xk.swap_dims(1, 2);
        let xv = xv.swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        let attn  = softmax(xq.matmul(xk.transpose()).mul_scalar(scale), 3);
        let out   = attn.matmul(xv);  // [1, H, S, Dh]

        self.wo.forward(out.swap_dims(1, 2).reshape([b, s, h * dh]))
    }
}
