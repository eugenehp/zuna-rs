/// Encoder Transformer with register interleaving (burn 0.20.1)
///
/// Python: `EncoderTransformer` in transformer.py.
///
/// Inference path (downsample_factor=1, bottleneck="mmd"):
///   1. For every input token, prepend a register token  →  [1, 2S, input_dim]
///   2. Embed with tok_embeddings                        →  [1, 2S, dim]
///   3. Run 16 encoder TransformerBlocks
///   4. Extract only the register tokens (position 0 of each pair) → [1, S, dim]
///   5. Norm + output projection                         →  [1, S, output_dim]
///   6. Bottleneck: "mmd" = passthrough at inference
use burn::prelude::*;
use burn::module::{Param, ParamId};
use burn::nn::{Linear, LinearConfig};
use crate::model::norm::RMSNorm;
use crate::model::encoder_block::EncoderBlock;
use crate::model::rope::RotaryEmbedding;

#[derive(Module, Debug)]
pub struct EncoderTransformer<B: Backend> {
    pub tok_embeddings: Linear<B>,
    /// Learnable register token prototype, shape [1, input_dim].
    pub registers: Param<Tensor<B, 2>>,
    pub layers:    Vec<EncoderBlock<B>>,
    pub norm:      RMSNorm<B>,
    pub output:    Linear<B>,
    pub downsample_factor: usize,
}

impl<B: Backend> EncoderTransformer<B> {
    pub fn new(
        input_dim:  usize,  // 32
        output_dim: usize,  // 32
        dim:        usize,  // 1024
        n_layers:   usize,  // 16
        head_dim:   usize,
        n_heads:    usize,
        n_kv_heads: usize,
        hidden_dim: usize,
        norm_eps:   f64,
        downsample_factor: usize,  // 1
        device:     &B::Device,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| EncoderBlock::new(
                dim, head_dim, n_heads, n_kv_heads, hidden_dim, norm_eps, device,
            ))
            .collect();
        Self {
            tok_embeddings: LinearConfig::new(input_dim, dim)
                .with_bias(true).init(device),
            registers: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, input_dim], device),
            ),
            layers,
            norm:   RMSNorm::new(dim, norm_eps, device),
            output: LinearConfig::new(dim, output_dim).with_bias(false).init(device),
            downsample_factor,
        }
    }

    /// token_values: [1, S, input_dim]   (zeroed channels = dropped)
    /// tok_idx:      [S, 4]              (discrete x,y,z,tc per token)
    /// Returns:      [1, S, output_dim]  (encoder latent)
    pub fn forward(
        &self,
        token_values: Tensor<B, 3>,
        tok_idx:      Tensor<B, 2, Int>,
        rope:         &RotaryEmbedding<B>,
    ) -> Tensor<B, 3> {
        let [b, s, d] = token_values.dims();
        let df        = self.downsample_factor; // 1

        // ── 1. Interleave register + real tokens ────────────────────────────
        // With df=1 layout is [reg_0, tok_0, reg_1, tok_1, …] → [1, 2S, d]
        let regs = self.registers
            .val()                          // [1, d]
            .unsqueeze_dim::<3>(0)          // [1, 1, d]
            .expand([b, s, d]);             // [1, S, d]

        // Stack [1,S,d] tensors along new dim 2 → [1, S, 2, d] → [1, 2S, d]
        // Tensor::stack::<4> takes 3-D inputs and produces 4-D output.
        let interleaved = Tensor::stack::<4>(vec![regs, token_values], 2)
            .reshape([b, s * (df + 1), d]);

        // ── 2. Embed ─────────────────────────────────────────────────────────
        let mut h = self.tok_embeddings.forward(interleaved); // [1, 2S, dim]

        // ── 3. Repeat tok_idx for the doubled sequence ───────────────────────
        // Python: tok_idx.repeat_interleave(repeats=2, dim=1)
        // Each position appears twice: once for register, once for real token.
        let tok_idx_2x = repeat_interleave_rows(tok_idx, 2); // [2S, 4]

        // ── 4. Build 4-D RoPE freqs ──────────────────────────────────────────
        let freqs = rope.build_freqs_4d(tok_idx_2x); // [2S, head_dim/2, 2, 2]

        // ── 5. Transformer layers ─────────────────────────────────────────────
        for layer in &self.layers {
            h = layer.forward(h, freqs.clone());
        }

        // ── 6. Extract register tokens ────────────────────────────────────────
        // With df=1: layout is [reg, tok, reg, tok, …].
        // Registers sit at positions {0, 2, 4, …} = index 0 within each pair.
        // Reshape [1, 2S, dim] → [1, S, 2, dim], take first of the 2.
        let hdim = h.dims()[2];
        let registers = h
            .reshape([b, s, df + 1, hdim])   // [1, S, 2, dim]
            .narrow(2, 0, 1)                  // [1, S, 1, dim]
            .reshape([b, s, hdim]);           // [1, S, dim]

        // ── 7. Output projection (bottleneck is passthrough at inference) ─────
        self.output.forward(self.norm.forward(registers)) // [1, S, output_dim]
    }
}

/// Repeat each row `repeats` times: [S, C] → [S*repeats, C]
/// Equivalent to PyTorch `repeat_interleave(t, repeats, dim=0)`.
fn repeat_interleave_rows<B: Backend>(
    t:       Tensor<B, 2, Int>,
    repeats: usize,
) -> Tensor<B, 2, Int> {
    let [s, c] = t.dims();
    // [S, C] → [S, 1, C] → expand [S, repeats, C] → [S*repeats, C]
    t.unsqueeze_dim::<3>(1)
        .expand([s, repeats, c])
        .reshape([s * repeats, c])
}
