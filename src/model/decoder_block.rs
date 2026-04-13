/// Decoder Block: cross-attention → self-attention → FFN (burn 0.20.1)
///
/// Python: `DecoderBlock` in xattn.py.
///   x   = x + XAttn(AdaRMSNorm(x,c), AdaRMSNorm(enc,c), freqs)
///   h   = x + SelfAttn(AdaRMSNorm(x,c), freqs)
///   out = h + FFN(AdaRMSNorm(h,c))
use burn::prelude::*;
use crate::model::norm::AdaRMSNorm;
use crate::model::attention::Attention;
use crate::model::cross_attention::CrossAttention;
use crate::model::feedforward::FeedForward;

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub cross_attention_x_norm: AdaRMSNorm<B>,
    pub cross_attention_y_norm: AdaRMSNorm<B>,
    pub cross_attention:        CrossAttention<B>,
    pub attention_norm:         AdaRMSNorm<B>,
    pub attention:              Attention<B>,
    pub ffn_norm:               AdaRMSNorm<B>,
    pub feed_forward:           FeedForward<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        dim:        usize,
        t_dim:      usize,
        head_dim:   usize,
        n_heads:    usize,
        n_kv_heads: usize,
        hidden_dim: usize,
        norm_eps:   f64,
        device:     &B::Device,
    ) -> Self {
        Self {
            cross_attention_x_norm: AdaRMSNorm::new(t_dim, dim, norm_eps, device),
            cross_attention_y_norm: AdaRMSNorm::new(t_dim, dim, norm_eps, device),
            cross_attention:        CrossAttention::new(dim, head_dim, n_heads, n_kv_heads, device),
            attention_norm:         AdaRMSNorm::new(t_dim, dim, norm_eps, device),
            attention:              Attention::new(dim, head_dim, n_heads, n_kv_heads, device),
            ffn_norm:               AdaRMSNorm::new(t_dim, dim, norm_eps, device),
            feed_forward:           FeedForward::new(dim, hidden_dim, device),
        }
    }

    /// x:        [1, S_q,  dim]  — decoder state
    /// y:        [1, S_kv, dim]  — encoder output (projected to dim)
    /// c:        [1, 1,    t_dim]— timestep embedding
    /// freqs_q:  [S_q,  head_dim/2, 2, 2]
    /// freqs_kv: [S_kv, head_dim/2, 2, 2]   (same as freqs_q for CR=1)
    pub fn forward(
        &self,
        x:        Tensor<B, 3>,
        y:        Tensor<B, 3>,
        c:        Tensor<B, 3>,
        freqs_q:  Tensor<B, 4>,
        freqs_kv: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        // Cross-attention residual
        let x_normed = self.cross_attention_x_norm.forward(x.clone(), c.clone());
        let y_normed = self.cross_attention_y_norm.forward(y, c.clone());
        let x = x + self.cross_attention.forward(x_normed, y_normed, freqs_q.clone(), freqs_kv);

        // Self-attention residual
        let h = x.clone()
            + self.attention.forward(
                self.attention_norm.forward(x.clone(), c.clone()),
                freqs_q.clone(),
            );

        // FFN residual
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h, c))
    }
}
