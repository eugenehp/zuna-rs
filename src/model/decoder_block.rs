//! Decoder Block: cross-attention → self-attention → FFN (burn 0.20.1)
//!
//! Python: `DecoderBlock` in xattn.py.
//!   x   = x + XAttnPost(XAttn(AdaRMSNorm(x,c), AdaRMSNorm(enc,c), freqs))
//!   h   = x + AttnPost(SelfAttn(AdaRMSNorm(x,c), freqs))
//!   out = h + FfnPost(FFN(AdaRMSNorm(h,c)))
//!
//! The `*Post` sandwich norms exist only in ZUNA1.1 (`do_sandwich_norm`);
//! on ZUNA1 they are `None` and the block reduces to plain pre-norm.
use crate::model::attention::Attention;
use crate::model::cross_attention::CrossAttention;
use crate::model::encoder_block::apply_post;
use crate::model::feedforward::FeedForward;
use crate::model::norm::{AdaRMSNorm, RMSNorm};
use crate::model::BlockDims;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub cross_attention_x_norm: AdaRMSNorm<B>,
    pub cross_attention_y_norm: AdaRMSNorm<B>,
    pub cross_attention: CrossAttention<B>,
    pub cross_attention_norm_post: Option<RMSNorm<B>>,
    pub attention_norm: AdaRMSNorm<B>,
    pub attention: Attention<B>,
    pub attention_norm_post: Option<RMSNorm<B>>,
    pub ffn_norm: AdaRMSNorm<B>,
    pub feed_forward: FeedForward<B>,
    pub ffn_norm_post: Option<RMSNorm<B>>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(d: BlockDims, t_dim: usize, device: &B::Device) -> Self {
        let post = || {
            d.arch
                .sandwich_norm
                .then(|| RMSNorm::new(d.dim, d.norm_eps, device))
        };
        let ada = || AdaRMSNorm::new(t_dim, d.dim, d.norm_eps, device);
        Self {
            cross_attention_x_norm: ada(),
            cross_attention_y_norm: ada(),
            cross_attention: CrossAttention::new(
                d.dim,
                d.head_dim,
                d.n_heads,
                d.n_kv_heads,
                d.arch.qk_norm,
                device,
            ),
            cross_attention_norm_post: post(),
            attention_norm: ada(),
            attention: Attention::new(
                d.dim,
                d.head_dim,
                d.n_heads,
                d.n_kv_heads,
                d.arch.qk_norm,
                device,
            ),
            attention_norm_post: post(),
            ffn_norm: ada(),
            feed_forward: FeedForward::new(d.dim, d.hidden_dim, device),
            ffn_norm_post: post(),
        }
    }

    /// x:        [1, S_q,  dim]  — decoder state
    /// y:        [1, S_kv, dim]  — encoder output (projected to dim)
    /// c:        [1, 1,    t_dim]— timestep embedding
    /// freqs_q:  [S_q,  head_dim/2, 2, 2]
    /// freqs_kv: [S_kv, head_dim/2, 2, 2]   (same as freqs_q for CR=1)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        y: Tensor<B, 3>,
        c: Tensor<B, 3>,
        freqs_q: Tensor<B, 4>,
        freqs_kv: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        // Cross-attention residual
        let x_normed = self.cross_attention_x_norm.forward(x.clone(), c.clone());
        let y_normed = self.cross_attention_y_norm.forward(y, c.clone());
        let xa = self
            .cross_attention
            .forward(x_normed, y_normed, freqs_q.clone(), freqs_kv);
        let x = x + apply_post(&self.cross_attention_norm_post, xa);

        // Self-attention residual
        let sa = self
            .attention
            .forward(self.attention_norm.forward(x.clone(), c.clone()), freqs_q);
        let h = x + apply_post(&self.attention_norm_post, sa);

        // FFN residual
        let ff = self
            .feed_forward
            .forward(self.ffn_norm.forward(h.clone(), c));
        h + apply_post(&self.ffn_norm_post, ff)
    }
}
