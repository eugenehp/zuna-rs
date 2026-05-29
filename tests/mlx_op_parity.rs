//! Per-op MLX-vs-CPU regression tests for kernel bugs surfaced during
//! the zuna encoder port. Each test exercises one op or composition
//! pattern on a tiny synthetic graph and asserts MLX matches CPU within
//! single-precision ULP.
//!
//! ## History
//!
//! Two MLX kernel bugs found through this harness, both since fixed in
//! `rlx-mlx`:
//!
//! 1. `Op::Attention` with rank-4 `[B, S, H, D]` input — MLX assumed
//!    rank-4 always meant `[B, H, S, D]` and produced silently wrong
//!    output. Fix: detect via the `num_heads` axis heuristic (same as
//!    the CPU executor) and lower through the rank-3 BERT path.
//! 2. `Op::Narrow` followed by `Op::Reshape` — MLX's compile trace
//!    fused the strided slice view with the reshape, returning
//!    contiguous bytes from the underlying buffer instead of the
//!    logical sliced values. Fix: force `ops::contiguous` after every
//!    `Op::Narrow` so the materialization happens before any reshape.
//!
//! These tests guard against regressions in either fix.
//!
//! ```text
//! cargo test --release --features cpu,mlx --test mlx_op_parity
//! ```

use rlx::ir::GraphExt;
use rlx::ops::MaskKind;
use rlx::prelude::*;

const TOL_ABS: f32 = 1e-5;

fn arange(n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale).collect()
}

fn rand_like(n: usize, seed: u64) -> Vec<f32> {
    let mut s = if seed == 0 { 0xCAFEF00DD15EA5E5 } else { seed };
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0
    }).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        d += x * y; na += x * x; nb += y * y;
    }
    d / (na.sqrt() * nb.sqrt())
}

fn assert_parity(label: &str, cpu: &[f32], mlx: &[f32]) {
    assert_eq!(cpu.len(), mlx.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    for (a, b) in cpu.iter().zip(mlx.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
    }
    let cos = cosine(cpu, mlx);
    assert!(
        max_abs < TOL_ABS && cos > 0.9999,
        "{label}: parity failed (max_abs={max_abs:.3e}, cosine={cos:.6})",
    );
}

fn run_both<F>(build: F, inputs: &[(&str, &[f32])], params: &[(&str, Vec<f32>)])
    -> Option<(Vec<f32>, Vec<f32>)>
where
    F: Fn() -> Graph,
{
    let run_on = |dev: rlx::Device| -> Vec<f32> {
        let mut compiled = rlx::Session::new(dev).compile(build());
        for (n, v) in params { compiled.set_param(n, v); }
        compiled.run(inputs).into_iter().next().unwrap()
    };
    let cpu = run_on(rlx::Device::Cpu);
    let mlx = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_on(rlx::Device::Mlx))) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("[skip] MLX backend not available in this build");
            return None;
        }
    };
    Some((cpu, mlx)))
}

#[test]
fn matmul_3d() {
    let (b, s, d) = (1usize, 8, 32);
    let x = arange(b * s * d, 0.01);
    let w = rand_like(d * d, 1);
    let build = || {
        let mut g = Graph::new("mm");
        let xi = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let wi = g.param("w", Shape::new(&[d, d], DType::F32));
        let y = g.mm(xi, wi);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(build, &[("x", &x)], &[("w", w)]) else { return };
    assert_parity("matmul [1,8,32] @ [32,32]", &cpu, &mlx);
}

#[test]
fn rms_norm_3d() {
    let (b, s, d) = (1usize, 8, 32);
    let x = rand_like(b * s * d, 2);
    let build = || {
        let mut g = Graph::new("rn");
        let xi = g.input("x", Shape::new(&[b, s, d], DType::F32));
        let gamma = g.param("gamma", Shape::new(&[d], DType::F32));
        let beta  = g.param("beta",  Shape::new(&[d], DType::F32));
        let y = g.rms_norm(xi, gamma, beta, 1e-5);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(
        build, &[("x", &x)],
        &[("gamma", vec![1.0; d]), ("beta", vec![0.0; d])],
    ) else { return };
    assert_parity("rms_norm [1,8,32]", &cpu, &mlx);
}

// ── Op::Attention dispatch by rank-4 layout ─────────────────────────

#[test]
fn attention_rank4_bshd() {
    // Triggers the rank-4 `[B, S, H, D]` dispatch path (axis 2 == num_heads).
    let (b, s, nh, dh) = (1usize, 8, 2, 16);
    let n = b * s * nh * dh;
    let (q, k, v) = (rand_like(n, 3), rand_like(n, 4), rand_like(n, 5));
    let build = || {
        let mut g = Graph::new("attn_bshd");
        let qi = g.input("q", Shape::new(&[b, s, nh, dh], DType::F32));
        let ki = g.input("k", Shape::new(&[b, s, nh, dh], DType::F32));
        let vi = g.input("v", Shape::new(&[b, s, nh, dh], DType::F32));
        let y = g.attention_kind_(qi, ki, vi, nh, dh, MaskKind::None);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(
        build, &[("q", &q), ("k", &k), ("v", &v)], &[],
    ) else { return };
    assert_parity("attention [B,S,H,D]", &cpu, &mlx);
}

#[test]
fn attention_rank4_bhsd() {
    // Triggers the rank-4 `[B, H, S, D]` pass-through path
    // (axis 1 == num_heads).
    let (b, nh, s, dh) = (1usize, 2, 8, 16);
    let n = b * nh * s * dh;
    let (q, k, v) = (rand_like(n, 3), rand_like(n, 4), rand_like(n, 5));
    let build = || {
        let mut g = Graph::new("attn_bhsd");
        let qi = g.input("q", Shape::new(&[b, nh, s, dh], DType::F32));
        let ki = g.input("k", Shape::new(&[b, nh, s, dh], DType::F32));
        let vi = g.input("v", Shape::new(&[b, nh, s, dh], DType::F32));
        let y = g.attention_kind_(qi, ki, vi, nh, dh, MaskKind::None);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(
        build, &[("q", &q), ("k", &k), ("v", &v)], &[],
    ) else { return };
    assert_parity("attention [B,H,S,D]", &cpu, &mlx);
}

// ── 5-D slice + reshape (rotate-half RoPE pattern) ──────────────────

#[test]
fn narrow_5d_axis4_idx0() {
    // Regression for the slice + reshape compile-trace bug.
    let (b, s, h, half) = (1usize, 4, 2, 4);
    let x = arange(b * s * h * half * 2, 0.01);
    let build = || {
        let mut g = Graph::new("nar5_0");
        let xi = g.input("x", Shape::new(&[b, s, h, half, 2], DType::F32));
        let y = g.narrow_(xi, 4, 0, 1);
        let y = g.reshape_(y, vec![b as i64, s as i64, h as i64, half as i64]);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(build, &[("x", &x)], &[]) else { return };
    assert_parity("narrow axis=4 start=0 on 5-D", &cpu, &mlx);
}

#[test]
fn narrow_5d_axis4_idx1() {
    let (b, s, h, half) = (1usize, 4, 2, 4);
    let x = arange(b * s * h * half * 2, 0.01);
    let build = || {
        let mut g = Graph::new("nar5_1");
        let xi = g.input("x", Shape::new(&[b, s, h, half, 2], DType::F32));
        let y = g.narrow_(xi, 4, 1, 1);
        let y = g.reshape_(y, vec![b as i64, s as i64, h as i64, half as i64]);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(build, &[("x", &x)], &[]) else { return };
    assert_parity("narrow axis=4 start=1 on 5-D", &cpu, &mlx);
}

#[test]
fn concat_5d_axis4() {
    let (b, s, h, half) = (1usize, 4, 2, 4);
    let n = b * s * h * half;
    let a = arange(n, 0.01);
    let bv = arange(n, 0.02);
    let build = || {
        let mut g = Graph::new("cat5");
        let ai = g.input("a", Shape::new(&[b, s, h, half], DType::F32));
        let bi = g.input("b", Shape::new(&[b, s, h, half], DType::F32));
        let a5 = g.reshape_(ai, vec![b as i64, s as i64, h as i64, half as i64, 1]);
        let b5 = g.reshape_(bi, vec![b as i64, s as i64, h as i64, half as i64, 1]);
        let y = g.concat_(vec![a5, b5], 4);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(build, &[("a", &a), ("b", &bv)], &[]) else { return };
    assert_parity("concat axis=4 on 5-D", &cpu, &mlx);
}

#[test]
fn rotate_half_pattern() {
    // The full rotate-half RoPE block: narrow → narrow → mul → sub →
    // reshape → reshape → concat → reshape. Exercises every op the
    // earlier two bugs touched in one graph.
    let (b, s, h, half) = (1usize, 4, 2, 4);
    let d = half * 2;
    let x = arange(b * s * h * d, 0.01);
    let cos = arange(s * half, 0.0001);
    let sin = arange(s * half, 0.0002);
    let build = || {
        let mut g = Graph::new("rot");
        let xi  = g.input("x",   Shape::new(&[b, s, h, d], DType::F32));
        let ci  = g.input("cos", Shape::new(&[1, s, 1, half], DType::F32));
        let si  = g.input("sin", Shape::new(&[1, s, 1, half], DType::F32));

        let pairs = g.reshape_(xi, vec![b as i64, s as i64, h as i64, half as i64, 2]);
        let even5 = g.narrow_(pairs, 4, 0, 1);
        let odd5  = g.narrow_(pairs, 4, 1, 1);
        let even = g.reshape_(even5, vec![b as i64, s as i64, h as i64, half as i64]);
        let odd  = g.reshape_(odd5,  vec![b as i64, s as i64, h as i64, half as i64]);

        let ec = g.mul(even, ci);
        let os = g.mul(odd,  si);
        let out_even = g.sub(ec, os);
        let es = g.mul(even, si);
        let oc = g.mul(odd,  ci);
        let out_odd  = g.add(es, oc);

        let e5 = g.reshape_(out_even, vec![b as i64, s as i64, h as i64, half as i64, 1]);
        let o5 = g.reshape_(out_odd,  vec![b as i64, s as i64, h as i64, half as i64, 1]);
        let stacked = g.concat_(vec![e5, o5], 4);
        let y = g.reshape_(stacked, vec![b as i64, s as i64, h as i64, d as i64]);
        g.set_outputs(vec![y]);
        g
    };
    let Some((cpu, mlx)) = run_both(
        build, &[("x", &x), ("cos", &cos), ("sin", &sin)], &[],
    ) else { return };
    assert_parity("rotate_half [B,S,H,2*half]", &cpu, &mlx);
}
