//! Encoder Transformer Block (burn 0.20.1)
//!
//! Python: `TransformerBlock` in lingua/transformer.py.
//!   h   = x + AttnPost(Attn(RMSNorm(x), freqs))
//!   out = h + FfnPost(FFN(RMSNorm(h)))
//!
//! The `*Post` sandwich norms exist only in ZUNA1.1 (`do_sandwich_norm`);
//! on ZUNA1 they are `None` and the block reduces to plain pre-norm.
use crate::model::attention::Attention;
use crate::model::feedforward::FeedForward;
use crate::model::norm::RMSNorm;
use crate::model::BlockDims;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    pub attention_norm: RMSNorm<B>,
    pub attention: Attention<B>,
    pub attention_norm_post: Option<RMSNorm<B>>,
    pub ffn_norm: RMSNorm<B>,
    pub feed_forward: FeedForward<B>,
    pub ffn_norm_post: Option<RMSNorm<B>>,
}

impl<B: Backend> EncoderBlock<B> {
    pub fn new(d: BlockDims, device: &B::Device) -> Self {
        let post = || {
            d.arch
                .sandwich_norm
                .then(|| RMSNorm::new(d.dim, d.norm_eps, device))
        };
        Self {
            attention_norm: RMSNorm::new(d.dim, d.norm_eps, device),
            attention: Attention::new(
                d.dim,
                d.head_dim,
                d.n_heads,
                d.n_kv_heads,
                d.arch.qk_norm,
                device,
            ),
            attention_norm_post: post(),
            ffn_norm: RMSNorm::new(d.dim, d.norm_eps, device),
            feed_forward: FeedForward::new(d.dim, d.hidden_dim, device),
            ffn_norm_post: post(),
        }
    }

    /// x:        [B, S, dim]
    /// freqs_4d: [S, head_dim/2, 2, 2]  (broadcasts over B)
    /// Returns:  [B, S, dim]
    pub fn forward(&self, x: Tensor<B, 3>, freqs_4d: Tensor<B, 4>) -> Tensor<B, 3> {
        let attn = self
            .attention
            .forward(self.attention_norm.forward(x.clone()), freqs_4d);
        let h = x + apply_post(&self.attention_norm_post, attn);

        let ff = self.feed_forward.forward(self.ffn_norm.forward(h.clone()));
        h + apply_post(&self.ffn_norm_post, ff)
    }
}

/// Sandwich norm on a sub-layer output; identity when the checkpoint has none.
pub(crate) fn apply_post<B: Backend>(norm: &Option<RMSNorm<B>>, y: Tensor<B, 3>) -> Tensor<B, 3> {
    match norm {
        Some(n) => n.forward(y),
        None => y,
    }
}
