/// Decoder Transformer — flow-matching denoiser (burn 0.20.1)
///
/// Python: `DecoderTransformer` in transformer.py.
///
///   h  = tok_embeddings(z)          [1, S, dim]
///   t  = t_embedder(time_t)         [1, 1, t_dim]
///   y  = encoder_proj(enc_out)      [1, S, dim]
///   for DecoderBlock: h = block(h, y, t, freqs, freqs)
///   return output(AdaRMSNorm(h, t)) [1, S, input_dim]
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use crate::model::norm::AdaRMSNorm;
use crate::model::conditioner::FourierConditioner;
use crate::model::decoder_block::DecoderBlock;
use crate::model::rope::RotaryEmbedding;

#[derive(Module, Debug)]
pub struct DecoderTransformer<B: Backend> {
    pub tok_embeddings: Linear<B>,
    pub t_embedder:     FourierConditioner<B>,
    pub encoder_proj:   Linear<B>,
    pub layers:         Vec<DecoderBlock<B>>,
    pub norm:           AdaRMSNorm<B>,
    pub output:         Linear<B>,
}

impl<B: Backend> DecoderTransformer<B> {
    pub fn new(
        input_dim:    usize,  // 32
        encoder_dim:  usize,  // 32  (encoder_output_dim)
        dim:          usize,  // 1024
        t_dim:        usize,  // 64
        n_layers:     usize,  // 16
        head_dim:     usize,
        n_heads:      usize,
        n_kv_heads:   usize,
        hidden_dim:   usize,
        norm_eps:     f64,
        device:       &B::Device,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| DecoderBlock::new(
                dim, t_dim, head_dim, n_heads, n_kv_heads, hidden_dim, norm_eps, device,
            ))
            .collect();
        Self {
            tok_embeddings: LinearConfig::new(input_dim, dim).with_bias(true).init(device),
            t_embedder:     FourierConditioner::new(t_dim, device),
            // Python: encoder_proj = nn.Linear(encoder_output_dim, dim, bias=True)
            encoder_proj:   LinearConfig::new(encoder_dim, dim).with_bias(true).init(device),
            layers,
            norm:   AdaRMSNorm::new(t_dim, dim, norm_eps, device),
            output: LinearConfig::new(dim, input_dim).with_bias(false).init(device),
        }
    }

    /// z:       [1, S, input_dim]  — current noisy EEG tokens
    /// enc_out: [1, S, encoder_dim]— encoder latent
    /// time_t:  [1, 1, 1]          — scalar timestep in [0, 1]
    /// tok_idx: [S, 4]
    /// Returns: velocity [1, S, input_dim]
    pub fn forward(
        &self,
        z:       Tensor<B, 3>,
        enc_out: Tensor<B, 3>,
        time_t:  Tensor<B, 3>,
        tok_idx: Tensor<B, 2, Int>,
        rope:    &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.tok_embeddings.forward(z);      // [1, S, dim]
        let t     = self.t_embedder.forward(time_t);     // [1, 1, t_dim]
        let y     = self.encoder_proj.forward(enc_out);  // [1, S, dim]

        let freqs = rope.build_freqs_4d(tok_idx);        // [S, head_dim/2, 2, 2]

        for layer in &self.layers {
            h = layer.forward(h, y.clone(), t.clone(), freqs.clone(), freqs.clone());
        }

        self.output.forward(self.norm.forward(h, t))
    }
}
