/// 4-D Axial Rotary Position Embedding (burn 0.20.1)
///
/// Mirrors `precompute_freqs_cis` + `apply_rotary_emb` from lingua/transformer.py,
/// extended to the 4-D case in transformer.py `Attention.forward` (rope_dim == 4).
///
/// Pretrained ZUNA hyperparameters:
///   rope_dim   = 4        (axes: x, y, z, t_coarse)
///   head_dim   = 64
///   max_seqlen = 50
///   rope_theta = 10_000.0
///
/// freqs_cis table shape: [max_seqlen, 8, 2, 2]
///   8  = (head_dim / rope_dim) / 2  = (64/4)/2
///   last two dims = 2×2 rotation matrix [[cos,-sin],[sin,cos]]

use burn::prelude::*;

pub struct RotaryEmbedding<B: Backend> {
    /// Shape: [max_seqlen, freqs_half_per_dim, 2, 2]
    pub freqs_cis: Tensor<B, 4>,
    pub max_seqlen: usize,
    pub rope_dim: usize,
    pub head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Precompute the freqs_cis rotation-matrix table.
    ///
    /// Python:
    ///   freqs = 1.0 / (theta ** (arange(0,dim,2)[:dim//2] / dim))
    ///   t     = arange(end)
    ///   freqs = outer(t, freqs)   # [end, dim//2]
    ///   cos, sin = freqs.cos(), freqs.sin()
    ///   return stack((cos,-sin,sin,cos),-1).view(*freqs.size(),2,2)
    pub fn new(
        head_dim: usize,
        rope_dim: usize,
        max_seqlen: usize,
        theta: f64,
        device: &B::Device,
    ) -> Self {
        assert_eq!(head_dim % rope_dim, 0);
        let dim_per_rope = head_dim / rope_dim; // 16
        let half = dim_per_rope / 2;            // 8

        // Build flat [max_seqlen * half * 4] rotation-matrix data.
        // Layout: [pos, h, row, col]  (row-major)
        let mut table = vec![0f32; max_seqlen * half * 4];
        for pos in 0..max_seqlen {
            for h in 0..half {
                let freq = 1.0 / theta.powf((2 * h) as f64 / dim_per_rope as f64) as f32;
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                let base = (pos * half + h) * 4;
                table[base]     =  c;  // [0,0]
                table[base + 1] = -s;  // [0,1]
                table[base + 2] =  s;  // [1,0]
                table[base + 3] =  c;  // [1,1]
            }
        }

        let freqs_cis = Tensor::<B, 1>::from_data(
            TensorData::new(table, vec![max_seqlen * half * 4]),
            device,
        )
        .reshape([max_seqlen, half, 2, 2]);

        Self { freqs_cis, max_seqlen, rope_dim, head_dim }
    }

    /// Gather rotation matrices for one RoPE axis.
    /// tok_idx_1d: [S] int tensor (values in 0..max_seqlen)
    /// Returns:    [S, half, 2, 2]
    fn gather_axis(&self, tok_idx_1d: Tensor<B, 1, Int>) -> Tensor<B, 4> {
        self.freqs_cis.clone().select(0, tok_idx_1d)
    }

    /// Build the combined 4-D freq tensor for a batch of tokens.
    ///
    /// Python (rope_dim==4):
    ///   parts = [freq_cis[tok_idx[:,i]] for i in range(4)]
    ///   freqcis_4RoPE = cat(parts, dim=1)   # [S, head_dim/2, 2, 2]
    ///
    /// tok_idx: [S, 4]  — one column per RoPE axis
    /// Returns: [S, head_dim/2, 2, 2]   (= [S, 32, 2, 2] for head_dim=64)
    pub fn build_freqs_4d(&self, tok_idx: Tensor<B, 2, Int>) -> Tensor<B, 4> {
        let s    = tok_idx.dims()[0];
        let _half = self.freqs_cis.dims()[1]; // 8

        let parts: Vec<Tensor<B, 4>> = (0..self.rope_dim)
            .map(|axis| {
                let col = tok_idx
                    .clone()
                    .narrow(1, axis, 1)   // [S, 1]
                    .reshape([s]);        // [S]
                self.gather_axis(col)     // [S, half, 2, 2]
            })
            .collect();

        // cat along dim 1: rope_dim × [S, half, 2, 2]  →  [S, rope_dim*half, 2, 2]
        Tensor::cat(parts, 1)
    }
}

/// Apply 4-D RoPE to query and key tensors.
///
/// Implements `apply_rotary_emb` from lingua/transformer.py using the
/// standard "rotate half" formulation:
///   x_even' =  x_even * cos - x_odd * sin
///   x_odd'  =  x_even * sin + x_odd * cos
///
/// xq, xk : [1, S, H, D]
/// freqs   : [S, D/2, 2, 2]
///
/// Returns rotated (xq', xk') with the same shape.
pub fn apply_rope<B: Backend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    freqs: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [_b, s, h, d] = xq.dims();
    let half = d / 2;

    // cos = freqs[..., 0, 0], sin = freqs[..., 1, 0]
    // Both shape [S, D/2], broadcast to [1, S, 1, D/2]
    let cos = freqs
        .clone()
        .narrow(2, 0, 1)  // [S, D/2, 1, 2]
        .narrow(3, 0, 1)  // [S, D/2, 1, 1]
        .reshape([1, s, 1, half]);
    let sin = freqs
        .narrow(2, 1, 1)  // [S, D/2, 1, 2]  (row 1 = [sin, cos])
        .narrow(3, 0, 1)  // sin column
        .reshape([1, s, 1, half]);

    (
        rotate_half(xq, cos.clone(), sin.clone(), s, h, half),
        rotate_half(xk, cos, sin, s, h, half),
    )
}

/// Apply rotation to a single [1, S, H, D] tensor.
fn rotate_half<B: Backend>(
    x:    Tensor<B, 4>,  // [1, S, H, D]
    cos:  Tensor<B, 4>,  // [1, S, 1, D/2]
    sin:  Tensor<B, 4>,  // [1, S, 1, D/2]
    s:    usize,
    h:    usize,
    half: usize,
) -> Tensor<B, 4> {
    // Reshape to [1, S, H, D/2, 2] then split even/odd
    let pairs = x.reshape([1, s, h, half, 2]);
    let even = pairs.clone().narrow(4, 0, 1).reshape([1, s, h, half]);
    let odd  = pairs.narrow(4, 1, 1).reshape([1, s, h, half]);

    let out_even = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let out_odd  = even * sin + odd * cos;

    // Interleave: stack [1,S,H,half] × 2 along new dim 4
    //   → [1, S, H, half, 2] → reshape → [1, S, H, D]
    // Tensor::stack::<5> takes 4-D inputs and produces 5-D output.
    Tensor::stack::<5>(vec![out_even, out_odd], 4)
        .reshape([1, s, h, half * 2])
}
