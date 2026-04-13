/// Encoder Transformer Block (burn 0.20.1)
///
/// Python: `TransformerBlock` in lingua/transformer.py.
///   h   = x + Attn(RMSNorm(x), freqs)
///   out = h + FFN(RMSNorm(h))
use burn::prelude::*;
use crate::model::norm::RMSNorm;
use crate::model::attention::Attention;
use crate::model::feedforward::FeedForward;

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    pub attention_norm: RMSNorm<B>,
    pub attention:      Attention<B>,
    pub ffn_norm:       RMSNorm<B>,
    pub feed_forward:   FeedForward<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(
        dim:        usize,
        head_dim:   usize,
        n_heads:    usize,
        n_kv_heads: usize,
        hidden_dim: usize,
        norm_eps:   f64,
        device:     &B::Device,
    ) -> Self {
        Self {
            attention_norm: RMSNorm::new(dim, norm_eps, device),
            attention:      Attention::new(dim, head_dim, n_heads, n_kv_heads, device),
            ffn_norm:       RMSNorm::new(dim, norm_eps, device),
            feed_forward:   FeedForward::new(dim, hidden_dim, device),
        }
    }

    /// x:        [1, S, dim]
    /// freqs_4d: [S, head_dim/2, 2, 2]
    /// Returns:  [1, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>, freqs_4d: Tensor<B, 4>) -> Tensor<B, 3> {
        let h = x.clone()
            + self.attention.forward(self.attention_norm.forward(x.clone()), freqs_4d.clone());
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }
}
