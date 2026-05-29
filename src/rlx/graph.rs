//! ZUNA encoder + decoder expressed as RLX IR graphs.
//!
//! Each builder returns an [`rlx::Graph`] whose inputs are set per call and
//! whose parameters are declared by name. The companion module
//! [`super::weights`] loads safetensors into a `HashMap<String, Vec<f32>>`
//! and pushes those into a compiled graph via `set_param`.
//!
//! ## Why the model is split this way
//!
//! Burn's `Tensor<B,D>` carries shape information at runtime. RLX requires
//! shapes to be known when the graph is built. We therefore parameterise
//! the builders by the per-call `(batch, sequence_length)` pair and cache
//! one [`rlx::CompiledGraph`] per distinct shape (see [`super::encoder`]).
//!
//! ## RoPE
//!
//! RLX's `Op::Rope` implements NeoX-style RoPE (split-half formulation),
//! whereas the original Burn implementation uses the **interleaved
//! (even/odd pair)** formulation. To keep numeric parity we re-implement
//! `rotate_half` here with element-wise ops; the `cos`/`sin` tables are
//! precomputed on the CPU from `tok_idx` (see [`super::data`]).
//!
//! ## Register interleave
//!
//! The encoder prepends one learnable "register" token before every real
//! EEG token. Rather than express the broadcast + stack pattern in IR, we
//! pre-interleave on the CPU (using the loaded `encoder.registers` weight
//! plus the per-batch token values) and feed the result as the encoder's
//! `x` input.

use rlx::ir::GraphExt;
use rlx::ops::Activation;
use rlx::prelude::*;

// ── Param key registry ───────────────────────────────────────────────────────
//
// Auxiliary parameters introduced by the RLX graph that are NOT present in
// the safetensors file. The loader fills these with constant data after
// compiling each graph.

/// Per-graph "ones" vector of length `dim`, used as the `gamma` of every
/// RMSNorm-as-AdaRMSNorm reduction.
pub const KEY_ONES_DIM:  &str = "__zuna.ones_dim";
/// Per-graph "zeros" vector of length `dim`, used as the `beta` of every
/// RMSNorm-as-AdaRMSNorm reduction.
pub const KEY_ZEROS_DIM: &str = "__zuna.zeros_dim";
/// Single-element `[2π]` constant used by the Fourier conditioner.
pub const KEY_TWO_PI:    &str = "__zuna.two_pi";

// ── Shape specs ──────────────────────────────────────────────────────────────

/// Architecture parameters required to build the encoder graph for one
/// `(batch, sequence_length)` shape.
#[derive(Clone, Copy, Debug)]
pub struct EncoderSpec {
    /// Batch dimension (1 for single-recording inference).
    pub b: usize,
    /// Logical (un-interleaved) sequence length — `n_channels × tc`.
    pub s: usize,
    /// Interleaved sequence length — `s * (1 + downsample_factor)`.
    pub s2: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub downsample_factor: usize,
    pub norm_eps: f32,
}

/// Architecture parameters required to build the decoder graph for one
/// `(batch, sequence_length)` shape.
#[derive(Clone, Copy, Debug)]
pub struct DecoderSpec {
    pub b: usize,
    pub s: usize,
    pub input_dim: usize,
    pub encoder_dim: usize,
    pub dim: usize,
    pub t_dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub hidden_dim: usize,
    pub norm_eps: f32,
}

// ── Shape helpers ────────────────────────────────────────────────────────────

fn s1(d: usize)                             -> Shape { Shape::new(&[d],             DType::F32) }
fn s2_(a: usize, b: usize)                  -> Shape { Shape::new(&[a, b],          DType::F32) }
fn s3(a: usize, b: usize, c: usize)         -> Shape { Shape::new(&[a, b, c],       DType::F32) }
fn s4(a: usize, b: usize, c: usize, d: usize) -> Shape { Shape::new(&[a, b, c, d], DType::F32) }

// ── Building blocks ──────────────────────────────────────────────────────────

/// Interleaved-pair RoPE applied to a `[B, S, H, D]` tensor.
///
/// `cos` and `sin` are `[1, S, 1, D/2]` and broadcast across `B` and `H`.
/// Implements:
/// ```text
/// even' = even * cos - odd * sin
/// odd'  = even * sin + odd * cos
/// ```
/// with pairs formed by splitting the last axis into `(D/2, 2)` and
/// re-interleaving afterwards.
fn rotate_half(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId,
    sin: NodeId,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
) -> NodeId {
    let half = d / 2;
    // x [B,S,H,D] → [B,S,H,half,2]
    let pairs = g.reshape_(x, vec![b as i64, s as i64, h as i64, half as i64, 2]);

    // narrow axis 4: even = idx 0, odd = idx 1 — each [B,S,H,half,1]
    let even5 = g.narrow_(pairs, 4, 0, 1);
    let odd5  = g.narrow_(pairs, 4, 1, 1);
    // squeeze trailing 1 → [B,S,H,half]
    let even  = g.reshape_(even5, vec![b as i64, s as i64, h as i64, half as i64]);
    let odd   = g.reshape_(odd5,  vec![b as i64, s as i64, h as i64, half as i64]);

    let ec = g.mul(even, cos);
    let os = g.mul(odd,  sin);
    let out_even = g.sub(ec, os);

    let es = g.mul(even, sin);
    let oc = g.mul(odd,  cos);
    let out_odd  = g.add(es, oc);

    // Re-interleave: reshape each to [B,S,H,half,1], concat axis 4 → [B,S,H,half,2]
    let e5 = g.reshape_(out_even, vec![b as i64, s as i64, h as i64, half as i64, 1]);
    let o5 = g.reshape_(out_odd,  vec![b as i64, s as i64, h as i64, half as i64, 1]);
    let stacked = g.concat_(vec![e5, o5], 4);
    g.reshape_(stacked, vec![b as i64, s as i64, h as i64, d as i64])
}

/// Scaled dot-product attention with rotate-half RoPE pre-applied to Q/K.
///
/// All Q, K and V come from the same source `x`. For cross-attention,
/// see [`cross_attention`].
fn self_attention(
    g: &mut Graph,
    x: NodeId,
    wq: NodeId, wk: NodeId, wv: NodeId, wo: NodeId,
    cos: NodeId, sin: NodeId,
    b: usize, s: usize, _d: usize, nh: usize, dh: usize,
) -> NodeId {
    let h_total = nh * dh;

    let q = g.mm(x, wq);
    let k = g.mm(x, wk);
    let v = g.mm(x, wv);

    let q4 = g.reshape_(q, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, nh as i64, dh as i64]);

    let q_rot = rotate_half(g, q4, cos, sin, b, s, nh, dh);
    let k_rot = rotate_half(g, k4, cos, sin, b, s, nh, dh);

    let attn = g.attention_kind(
        q_rot, k_rot, v4,
        nh, dh,
        rlx::ops::MaskKind::None,
        s4(b, s, nh, dh),
    );
    let attn_3 = g.reshape_(attn, vec![b as i64, s as i64, h_total as i64]);
    g.mm(attn_3, wo)
}

/// Cross-attention: Q from `xq`, K/V from `xkv`. RoPE applied to both.
fn cross_attention(
    g: &mut Graph,
    xq: NodeId, xkv: NodeId,
    wq: NodeId, wk: NodeId, wv: NodeId, wo: NodeId,
    cos: NodeId, sin: NodeId,
    b: usize, s: usize, _d: usize, nh: usize, dh: usize,
) -> NodeId {
    let h_total = nh * dh;
    let q = g.mm(xq,  wq);
    let k = g.mm(xkv, wk);
    let v = g.mm(xkv, wv);

    let q4 = g.reshape_(q, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let k4 = g.reshape_(k, vec![b as i64, s as i64, nh as i64, dh as i64]);
    let v4 = g.reshape_(v, vec![b as i64, s as i64, nh as i64, dh as i64]);

    let q_rot = rotate_half(g, q4, cos, sin, b, s, nh, dh);
    let k_rot = rotate_half(g, k4, cos, sin, b, s, nh, dh);

    let attn = g.attention_kind(
        q_rot, k_rot, v4,
        nh, dh,
        rlx::ops::MaskKind::None,
        s4(b, s, nh, dh),
    );
    let attn_3 = g.reshape_(attn, vec![b as i64, s as i64, h_total as i64]);
    g.mm(attn_3, wo)
}

/// SwiGLU: `w2(silu(w1(x)) * w3(x))`.
fn swiglu_ffn(
    g: &mut Graph,
    x: NodeId,
    w1: NodeId, w2: NodeId, w3: NodeId,
) -> NodeId {
    let a = g.mm(x, w1);
    let act = g.silu(a);
    let c = g.mm(x, w3);
    let gated = g.mul(act, c);
    g.mm(gated, w2)
}

/// Adaptive RMS-norm: `x * rsqrt(mean(x², -1) + eps) * Linear(c)`.
///
/// `c` is `[B, 1, t_dim]`. The inner linear (named `proj_w`, `proj_b` in
/// the safetensors file) projects to `[B, 1, dim]` and broadcasts over
/// the sequence axis.
fn ada_rms_norm(
    g: &mut Graph,
    x: NodeId,
    c: NodeId,
    proj_w: NodeId, proj_b: NodeId,
    ones_dim: NodeId, zeros_dim: NodeId,
    eps: f32,
) -> NodeId {
    // First normalise with unit gamma / zero beta — produces the same
    // result as the manual rsqrt formulation in the original Burn code.
    let normed = g.rms_norm(x, ones_dim, zeros_dim, eps);

    // Adaptive scale: Linear(c) → [B, 1, dim], broadcast over S.
    let lc = g.mm(c, proj_w);
    let cm = g.add(lc, proj_b);
    g.mul(normed, cm)
}

// ── Encoder block ────────────────────────────────────────────────────────────

fn encoder_block(
    g: &mut Graph,
    x: NodeId,
    cos: NodeId, sin: NodeId,
    spec: &EncoderSpec,
    layer_idx: usize,
) -> NodeId {
    let d  = spec.dim;
    let nh = spec.n_heads;
    let dh = spec.head_dim;
    let p  = format!("encoder.layers.{layer_idx}");

    let an_g = g.param(format!("{p}.attention_norm.weight"), s1(d));
    let zb   = g.param(KEY_ZEROS_DIM, s1(d));
    let xn   = g.rms_norm(x, an_g, zb, spec.norm_eps);

    let wq = g.param(format!("{p}.attention.wq.weight"), s2_(d, nh * dh));
    let wk = g.param(format!("{p}.attention.wk.weight"), s2_(d, nh * dh));
    let wv = g.param(format!("{p}.attention.wv.weight"), s2_(d, nh * dh));
    let wo = g.param(format!("{p}.attention.wo.weight"), s2_(nh * dh, d));

    let attn = self_attention(g, xn, wq, wk, wv, wo, cos, sin,
                              spec.b, spec.s2, d, nh, dh);
    let x = g.add(x, attn);

    let fn_g = g.param(format!("{p}.ffn_norm.weight"), s1(d));
    let zb2  = g.param(KEY_ZEROS_DIM, s1(d));
    let hn   = g.rms_norm(x, fn_g, zb2, spec.norm_eps);

    let w1 = g.param(format!("{p}.feed_forward.w1.weight"), s2_(d, spec.hidden_dim));
    let w2 = g.param(format!("{p}.feed_forward.w2.weight"), s2_(spec.hidden_dim, d));
    let w3 = g.param(format!("{p}.feed_forward.w3.weight"), s2_(d, spec.hidden_dim));

    let ff = swiglu_ffn(g, hn, w1, w2, w3);
    g.add(x, ff)
}

// ── Decoder block ────────────────────────────────────────────────────────────

fn decoder_block(
    g: &mut Graph,
    x: NodeId, y: NodeId, c: NodeId,
    cos: NodeId, sin: NodeId,
    spec: &DecoderSpec,
    layer_idx: usize,
) -> NodeId {
    let d  = spec.dim;
    let nh = spec.n_heads;
    let dh = spec.head_dim;
    let td = spec.t_dim;
    let p  = format!("decoder.layers.{layer_idx}");

    let ones  = g.param(KEY_ONES_DIM,  s1(d));
    let zeros = g.param(KEY_ZEROS_DIM, s1(d));

    // ── cross-attention ──
    let xn_w = g.param(format!("{p}.cross_attention_x_norm.weight.weight"), s2_(td, d));
    let xn_b = g.param(format!("{p}.cross_attention_x_norm.weight.bias"),   s1(d));
    let yn_w = g.param(format!("{p}.cross_attention_y_norm.weight.weight"), s2_(td, d));
    let yn_b = g.param(format!("{p}.cross_attention_y_norm.weight.bias"),   s1(d));
    let x_norm = ada_rms_norm(g, x, c, xn_w, xn_b, ones, zeros, spec.norm_eps);
    let y_norm = ada_rms_norm(g, y, c, yn_w, yn_b, ones, zeros, spec.norm_eps);

    let cwq = g.param(format!("{p}.cross_attention.wq.weight"), s2_(d, nh * dh));
    let cwk = g.param(format!("{p}.cross_attention.wk.weight"), s2_(d, nh * dh));
    let cwv = g.param(format!("{p}.cross_attention.wv.weight"), s2_(d, nh * dh));
    let cwo = g.param(format!("{p}.cross_attention.wo.weight"), s2_(nh * dh, d));
    let xa  = cross_attention(g, x_norm, y_norm, cwq, cwk, cwv, cwo, cos, sin,
                              spec.b, spec.s, d, nh, dh);
    let x = g.add(x, xa);

    // ── self-attention ──
    let an_w = g.param(format!("{p}.attention_norm.weight.weight"), s2_(td, d));
    let an_b = g.param(format!("{p}.attention_norm.weight.bias"),   s1(d));
    let xn   = ada_rms_norm(g, x, c, an_w, an_b, ones, zeros, spec.norm_eps);

    let wq = g.param(format!("{p}.attention.wq.weight"), s2_(d, nh * dh));
    let wk = g.param(format!("{p}.attention.wk.weight"), s2_(d, nh * dh));
    let wv = g.param(format!("{p}.attention.wv.weight"), s2_(d, nh * dh));
    let wo = g.param(format!("{p}.attention.wo.weight"), s2_(nh * dh, d));
    let sa = self_attention(g, xn, wq, wk, wv, wo, cos, sin,
                            spec.b, spec.s, d, nh, dh);
    let h = g.add(x, sa);

    // ── feed-forward ──
    let fn_w = g.param(format!("{p}.ffn_norm.weight.weight"), s2_(td, d));
    let fn_b = g.param(format!("{p}.ffn_norm.weight.bias"),   s1(d));
    let hn   = ada_rms_norm(g, h, c, fn_w, fn_b, ones, zeros, spec.norm_eps);

    let w1 = g.param(format!("{p}.feed_forward.w1.weight"), s2_(d, spec.hidden_dim));
    let w2 = g.param(format!("{p}.feed_forward.w2.weight"), s2_(spec.hidden_dim, d));
    let w3 = g.param(format!("{p}.feed_forward.w3.weight"), s2_(d, spec.hidden_dim));
    let ff = swiglu_ffn(g, hn, w1, w2, w3);
    g.add(h, ff)
}

// ── Top-level graph builders ─────────────────────────────────────────────────

/// Build the encoder graph for one `(b, s2)` shape.
///
/// Inputs (set via `compiled.run`):
/// * `x` — `[B, S2, input_dim]` (already register-interleaved on the CPU
///   side; see [`super::data::preinterleave`]).
/// * `freqs_cos`, `freqs_sin` — `[1, S2, 1, head_dim/2]` (precomputed from
///   `tok_idx`; see [`super::data::precompute_rope`]).
///
/// Output: `[B, S, output_dim]`.
pub fn build_encoder_graph(spec: &EncoderSpec) -> Graph {
    let mut g = Graph::new("zuna_encoder");

    let id = spec.input_dim;
    let od = spec.output_dim;
    let d  = spec.dim;
    let dh = spec.head_dim;
    let df = spec.downsample_factor;
    let b  = spec.b;
    let s  = spec.s;
    let s2 = spec.s2;

    let x   = g.input("x",         s3(b, s2, id));
    let cos = g.input("freqs_cos", s4(1, s2, 1, dh / 2));
    let sin = g.input("freqs_sin", s4(1, s2, 1, dh / 2));

    // Embedding: Linear(input_dim → dim) with bias.
    let emb_w = g.param("encoder.tok_embeddings.weight", s2_(id, d));
    let emb_b = g.param("encoder.tok_embeddings.bias",   s1(d));
    let h0    = g.mm(x, emb_w);
    let mut h = g.add(h0, emb_b);

    for i in 0..spec.n_layers {
        h = encoder_block(&mut g, h, cos, sin, spec, i);
    }

    // Final RMSNorm — note: encoder uses RmsNorm with single `gamma` only.
    let n_g = g.param("encoder.norm.weight", s1(d));
    let zb  = g.param(KEY_ZEROS_DIM, s1(d));
    let h   = g.rms_norm(h, n_g, zb, spec.norm_eps);

    // De-interleave: keep only the register positions. [B,S2,d] →
    // [B,S,df+1,d] → narrow axis 2 from 0 len 1 → [B,S,1,d] → [B,S,d].
    let h5 = g.reshape_(h, vec![b as i64, s as i64, (df + 1) as i64, d as i64]);
    let regs5 = g.narrow_(h5, 2, 0, 1);
    let regs  = g.reshape_(regs5, vec![b as i64, s as i64, d as i64]);

    let out_w = g.param("encoder.output.weight", s2_(d, od));
    let out   = g.mm(regs, out_w);

    g.set_outputs(vec![out]);
    g
}

/// Build the decoder graph for one diffusion step on a single `(b, s)` shape.
///
/// Inputs (set via `compiled.run`):
/// * `z`         — `[B, S, input_dim]` current noisy tokens.
/// * `enc_out`   — `[B, S, encoder_dim]` cached encoder latent.
/// * `time_t`    — `[B, 1, 1]` scalar timestep in `[0, 1]`.
/// * `freqs_cos`, `freqs_sin` — `[1, S, 1, head_dim/2]`.
///
/// Output: `[B, S, input_dim]` velocity vector.
pub fn build_decoder_graph(spec: &DecoderSpec) -> Graph {
    let mut g = Graph::new("zuna_decoder");

    let b  = spec.b;
    let s  = spec.s;
    let d  = spec.dim;
    let id = spec.input_dim;
    let ed = spec.encoder_dim;
    let td = spec.t_dim;
    let dh = spec.head_dim;

    let z         = g.input("z",         s3(b, s, id));
    let enc_out   = g.input("enc_out",   s3(b, s, ed));
    let time_t    = g.input("time_t",    s3(b, 1, 1));
    let cos       = g.input("freqs_cos", s4(1, s, 1, dh / 2));
    let sin       = g.input("freqs_sin", s4(1, s, 1, dh / 2));

    // Token embeddings: z → h [B, S, dim]
    let te_w = g.param("decoder.tok_embeddings.weight", s2_(id, d));
    let te_b = g.param("decoder.tok_embeddings.bias",   s1(d));
    let h0   = g.mm(z, te_w);
    let h    = g.add(h0, te_b);

    // FourierConditioner: weight stored on disk as [t_dim/2, 1]; we
    // pre-transpose at load time to [1, t_dim/2] so we can mm directly.
    let tw      = g.param("decoder.t_embedder.weight", s2_(1, td / 2));
    let two_pi  = g.param(KEY_TWO_PI, s1(1));
    let f       = g.mm(time_t, tw);
    let f_scaled = g.mul(f, two_pi);
    let s_cos = s3(b, 1, td / 2);
    let cos_f = g.activation(Activation::Cos, f_scaled, s_cos.clone());
    let sin_f = g.activation(Activation::Sin, f_scaled, s_cos);
    let cat   = g.concat_(vec![cos_f, sin_f], 2);

    let tp_w = g.param("decoder.t_embedder.proj.weight", s2_(td, td));
    let tp_b = g.param("decoder.t_embedder.proj.bias",   s1(td));
    let tm   = g.mm(cat, tp_w);
    let c    = g.add(tm, tp_b);

    // encoder_proj: enc_out → y [B,S,dim]
    let ep_w = g.param("decoder.encoder_proj.weight", s2_(ed, d));
    let ep_b = g.param("decoder.encoder_proj.bias",   s1(d));
    let ym   = g.mm(enc_out, ep_w);
    let y    = g.add(ym, ep_b);

    // Decoder layers
    let mut h = h;
    for i in 0..spec.n_layers {
        h = decoder_block(&mut g, h, y, c, cos, sin, spec, i);
    }

    // Final AdaRMSNorm + output linear.
    let ones  = g.param(KEY_ONES_DIM,  s1(d));
    let zeros = g.param(KEY_ZEROS_DIM, s1(d));
    let fn_w = g.param("decoder.norm.weight.weight", s2_(td, d));
    let fn_b = g.param("decoder.norm.weight.bias",   s1(d));
    let h    = ada_rms_norm(&mut g, h, c, fn_w, fn_b, ones, zeros, spec.norm_eps);

    let out_w = g.param("decoder.output.weight", s2_(d, id));
    let out   = g.mm(h, out_w);

    g.set_outputs(vec![out]);
    g
}

