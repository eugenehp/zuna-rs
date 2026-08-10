//! Decoder Transformer — flow-matching denoiser (burn 0.20.1)
//!
//! Python: `DecoderTransformer` in transformer.py.
//!
//!   h  = tok_embeddings(z)          [1, S, dim]
//!   t  = t_embedder(time_t)         [1, 1, t_dim]
//!   y  = encoder_proj(enc_out)      [1, S, dim]
//!   for DecoderBlock: h = block(h, y, t, freqs, freqs)
//!   return output(AdaRMSNorm(h, t)) [1, S, input_dim]
use crate::model::conditioner::FourierConditioner;
use crate::model::decoder_block::DecoderBlock;
use crate::model::norm::AdaRMSNorm;
use crate::model::rope::RotaryEmbedding;
use crate::model::{linear_zeros, DecoderDims};
use burn::nn::Linear;
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct DecoderTransformer<B: Backend> {
    pub tok_embeddings: Linear<B>,
    pub t_embedder: FourierConditioner<B>,
    pub encoder_proj: Linear<B>,
    pub layers: Vec<DecoderBlock<B>>,
    pub norm: AdaRMSNorm<B>,
    pub output: Linear<B>,
}

impl<B: Backend> DecoderTransformer<B> {
    pub fn new(d: DecoderDims, device: &B::Device) -> Self {
        let layers = (0..d.n_layers)
            .map(|_| DecoderBlock::new(d.block, d.t_dim, device))
            .collect();
        Self {
            tok_embeddings: linear_zeros(d.input_dim, d.block.dim, true, device),
            t_embedder: FourierConditioner::new(d.t_dim, device),
            encoder_proj: linear_zeros(d.encoder_dim, d.block.dim, true, device),
            layers,
            norm: AdaRMSNorm::new(d.t_dim, d.block.dim, d.block.norm_eps, device),
            output: linear_zeros(d.block.dim, d.input_dim, false, device),
        }
    }

    /// z:       [1, S, input_dim]  — current noisy EEG tokens
    /// enc_out: [1, S, encoder_dim]— encoder latent
    /// time_t:  [1, 1, 1]          — scalar timestep in [0, 1]
    /// tok_idx: [S, 4]
    /// Returns: velocity [1, S, input_dim]
    pub fn forward(
        &self,
        z: Tensor<B, 3>,
        enc_out: Tensor<B, 3>,
        time_t: Tensor<B, 3>,
        tok_idx: Tensor<B, 2, Int>,
        rope: &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.tok_embeddings.forward(z); // [1, S, dim]
        let t = self.t_embedder.forward(time_t); // [1, 1, t_dim]
        let y = self.encoder_proj.forward(enc_out); // [1, S, dim]

        let freqs = rope.build_freqs_4d(tok_idx); // [S, head_dim/2, 2, 2]

        for layer in &self.layers {
            h = layer.forward(h, y.clone(), t.clone(), freqs.clone(), freqs.clone());
        }

        self.output.forward(self.norm.forward(h, t))
    }
}
