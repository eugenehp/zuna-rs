/// Cross-Attention with 4-D axial RoPE (burn 0.20.1)
///
/// Python: `CrossAttention` in xattn.py.
/// Q comes from the decoder state, K/V from the encoder output.
/// Each side is rotated with its own freqs tensor.
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use crate::model::rope::apply_rope;

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    pub n_heads:    usize,
    pub n_kv_heads: usize,
    pub head_dim:   usize,
}

impl<B: Backend> CrossAttention<B> {
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

    /// xq:       [1, S_q,  dim]  — decoder state
    /// xkv:      [1, S_kv, dim]  — encoder output (already projected to `dim`)
    /// freqs_q:  [S_q,  head_dim/2, 2, 2]
    /// freqs_kv: [S_kv, head_dim/2, 2, 2]
    /// Returns:  [1, S_q, dim]
    pub fn forward(
        &self,
        xq:      Tensor<B, 3>,
        xkv:     Tensor<B, 3>,
        freqs_q:  Tensor<B, 4>,
        freqs_kv: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [b, s_q, _]  = xq.dims();
        let [_, s_kv, _] = xkv.dims();
        let (h, dh) = (self.n_heads, self.head_dim);
        let device  = xq.device();

        let q = self.wq.forward(xq).reshape([b, s_q,  h, dh]);
        let k = self.wk.forward(xkv.clone()).reshape([b, s_kv, h, dh]);
        let v = self.wv.forward(xkv).reshape([b, s_kv, h, dh]);

        // Rotate Q with freqs_q: use a zero dummy tensor for the K partner.
        let (q_rot, _) = apply_rope(
            q,
            Tensor::zeros([b, s_q, h, dh], &device),
            freqs_q,
        );
        // Rotate K with freqs_kv: use a zero dummy tensor for the Q partner.
        let (_, k_rot) = apply_rope(
            Tensor::zeros([b, s_kv, h, dh], &device),
            k,
            freqs_kv,
        );

        let q_t = q_rot.swap_dims(1, 2);  // [1, H, S_q,  Dh]
        let k_t = k_rot.swap_dims(1, 2);  // [1, H, S_kv, Dh]
        let v_t = v.swap_dims(1, 2);

        let scale = (dh as f64).powf(-0.5) as f32;
        // [1,H,S_q,Dh] × [1,H,Dh,S_kv] → [1,H,S_q,S_kv]
        let attn = softmax(q_t.matmul(k_t.transpose()).mul_scalar(scale), 3);
        // [1,H,S_q,S_kv] × [1,H,S_kv,Dh] → [1,H,S_q,Dh]
        let out = attn.matmul(v_t);

        self.wo.forward(out.swap_dims(1, 2).reshape([b, s_q, h * dh]))
    }
}
