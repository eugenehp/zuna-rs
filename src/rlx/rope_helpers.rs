//! CPU-side helpers that produce the inputs the RLX graphs expect.
//!
//! The original Burn implementation built the 4-axial RoPE freqs table
//! and the register interleave inside the model's forward pass using
//! gather/stack ops. RLX prefers tensor inputs of known shapes; we
//! therefore precompute these on the CPU before each `compiled.run`.

/// Build a 1-D RoPE rotation table, mirroring `RotaryEmbedding::new` in
/// the original Burn implementation.
///
/// Returns `[max_seqlen, half, 4]` row-major where each `[pos, h, :]`
/// slot stores `[cos, -sin, sin, cos]` exactly as the Burn code does.
/// The `half` count is `head_dim / rope_dim / 2`.
pub fn build_rope_table(
    head_dim: usize,
    rope_dim: usize,
    max_seqlen: usize,
    theta: f64,
) -> Vec<f32> {
    assert_eq!(head_dim % rope_dim, 0, "head_dim must be divisible by rope_dim");
    let dim_per_rope = head_dim / rope_dim;
    let half = dim_per_rope / 2;
    let mut table = vec![0f32; max_seqlen * half * 4];
    for pos in 0..max_seqlen {
        for h in 0..half {
            let freq = 1.0_f64 / theta.powf((2 * h) as f64 / dim_per_rope as f64);
            let angle = (pos as f64) * freq;
            let (s, c) = (angle as f32).sin_cos();
            let base = (pos * half + h) * 4;
            table[base]     = c;
            table[base + 1] = -s;
            table[base + 2] = s;
            table[base + 3] = c;
        }
    }
    table
}

/// Build the per-token `(cos, sin)` tensors for a sequence of length `s2`:
/// * `cos` — `[1, s2, 1, head_dim/2]`
/// * `sin` — `[1, s2, 1, head_dim/2]`
///
/// `tok_idx` is `[s2, 4]` row-major i32. `rope_dim` (= 4 for ZUNA)
/// determines how many axial RoPE blocks are concatenated.
pub fn precompute_rope(
    tok_idx: &[i32],
    rope_table: &[f32],
    head_dim: usize,
    rope_dim: usize,
    s2: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim_per_rope = head_dim / rope_dim;
    let half_per_axis = dim_per_rope / 2;
    let half_total = head_dim / 2;
    assert_eq!(half_total, rope_dim * half_per_axis);
    assert_eq!(tok_idx.len(), s2 * 4);

    let mut cos = vec![0f32; s2 * half_total];
    let mut sin = vec![0f32; s2 * half_total];

    for pos in 0..s2 {
        for axis in 0..rope_dim {
            let idx = tok_idx[pos * 4 + axis] as usize;
            for h in 0..half_per_axis {
                let table_base = (idx * half_per_axis + h) * 4;
                let c = rope_table[table_base];        // cos
                let neg_s = rope_table[table_base + 1]; // -sin
                let s_val = -neg_s;                     // sin
                let out_dim = axis * half_per_axis + h;
                cos[pos * half_total + out_dim] = c;
                sin[pos * half_total + out_dim] = s_val;
            }
        }
    }
    (cos, sin)
}

/// Interleave register tokens with real tokens on the CPU.
///
/// "Register prepend" step:
/// ```text
/// for each i in 0..s:
///   out[i*2]   = registers           // [input_dim]
///   out[i*2+1] = token_values[i, :]  // [input_dim]
/// ```
///
/// `token_values` is `[b, s, input_dim]` row-major; `registers` is
/// `[input_dim]`. Output is `[b, s2, input_dim]` row-major, where
/// `s2 = s * (1 + downsample_factor)`. The downsample_factor=1 case
/// (the only one ZUNA ships) gives `s2 = 2 * s`.
pub fn preinterleave(
    token_values: &[f32],
    registers:    &[f32],
    b: usize,
    s: usize,
    input_dim: usize,
    downsample_factor: usize,
) -> Vec<f32> {
    assert_eq!(token_values.len(), b * s * input_dim);
    assert_eq!(registers.len(), input_dim);
    let stride = downsample_factor + 1;
    let s2 = s * stride;
    let mut out = vec![0f32; b * s2 * input_dim];

    for bi in 0..b {
        for i in 0..s {
            // Slot 0 of the group: register token.
            let reg_off = ((bi * s2) + i * stride) * input_dim;
            out[reg_off .. reg_off + input_dim].copy_from_slice(registers);
            // Subsequent slots: the real token (and copies for df > 1).
            for k in 1..stride {
                let tok_off = ((bi * s2) + i * stride + k) * input_dim;
                let src_off = ((bi * s) + i) * input_dim;
                out[tok_off .. tok_off + input_dim]
                    .copy_from_slice(&token_values[src_off .. src_off + input_dim]);
            }
        }
    }
    out
}

/// Repeat each row `df + 1` times: `[s, 4]` → `[s * (df + 1), 4]`.
/// PyTorch equivalent: `tok_idx.repeat_interleave(repeats=df+1, dim=0)`.
pub fn repeat_token_idx(tok_idx: &[i32], s: usize, downsample_factor: usize) -> Vec<i32> {
    let stride = downsample_factor + 1;
    assert_eq!(tok_idx.len(), s * 4);
    let mut out = vec![0i32; s * stride * 4];
    for i in 0..s {
        for k in 0..stride {
            let dst = (i * stride + k) * 4;
            let src = i * 4;
            out[dst .. dst + 4].copy_from_slice(&tok_idx[src .. src + 4]);
        }
    }
    out
}

/// Sample standard Normal noise into a fresh `Vec<f32>` of `n` elements
/// using a deterministic xorshift64* seed. Reproducible across runs;
/// pair with a fixed `seed` when bit-comparing against another runtime.
pub fn sample_normal(n: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut s = if seed == 0 { 0xCAFEF00DD15EA5E5 } else { seed };
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        // Box–Muller using two uniform-in-(0,1] samples.
        let u1 = next_unit(&mut s);
        let u2 = next_unit(&mut s);
        let r = (-2.0_f64 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        out.push((z0 as f32) * sigma);
        i += 1;
        if i < n {
            out.push((z1 as f32) * sigma);
            i += 1;
        }
    }
    out
}

fn next_unit(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let mant = (*state >> 11) as f64 / ((1u64 << 53) as f64);
    // Avoid exactly 0 so ln() is finite.
    if mant <= 0.0 { f64::MIN_POSITIVE } else { mant }
}
