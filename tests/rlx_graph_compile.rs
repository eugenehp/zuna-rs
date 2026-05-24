//! IR-level tests for the RLX encoder / decoder graphs.
//!
//! No safetensors weights required: build the graph, compile it,
//! push zero-filled buffers as every parameter, and assert the runtime
//! produces an output of the expected shape. Catches op-name and
//! shape-builder typos that the parity tests can only surface when
//! real weights are available.

#![cfg(feature = "rlx-backend")]

use zuna_rs::rlx::data::{precompute_rope, build_rope_table, preinterleave, repeat_token_idx};
use zuna_rs::rlx::graph::{build_encoder_graph, build_decoder_graph,
                          EncoderSpec, DecoderSpec,
                          KEY_ONES_DIM, KEY_TWO_PI, KEY_ZEROS_DIM};

/// Push a buffer for every `Op::Param` declared in `graph`. The
/// `overrides` map provides specific values for named params
/// (e.g. the `__zuna.ones_dim` / `__zuna.zeros_dim` constants);
/// everything else gets a zero vector of the declared length.
///
/// Uses [`rlx::Graph::params`] (added as part of the rlx 0.2.x
/// introspection helpers) so we don't have to walk `nodes()` and
/// pattern-match on `Op::Param` by hand.
fn zero_fill_params(graph: &rlx::Graph, compiled: &mut rlx::CompiledGraph,
                    overrides: &[(&str, Vec<f32>)]) {
    let mut ovr: std::collections::HashMap<&str, Vec<f32>> =
        overrides.iter().map(|(k, v)| (*k, v.clone())).collect();
    for (name, shape, _id) in graph.params() {
        let n = shape.elem_count();
        let buf = ovr.remove(name).unwrap_or_else(|| vec![0.0; n]);
        assert_eq!(buf.len(), n, "param {name}: buffer len {} != shape {}", buf.len(), n);
        compiled.set_param(name, &buf);
    }
}

#[test]
fn encoder_graph_compiles_and_runs() {
    // Tiny synthetic shape — proves the IR builders are valid; not
    // intended to produce a meaningful EEG embedding.
    let spec = EncoderSpec {
        b: 1,
        s: 4,
        s2: 8,
        input_dim:  8,
        output_dim: 8,
        dim:        16,
        n_layers:   2,
        head_dim:   8,
        n_heads:    2,
        hidden_dim: 32,
        downsample_factor: 1,
        norm_eps:   1e-5,
    };

    let graph = build_encoder_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());

    zero_fill_params(&graph, &mut compiled, &[
        (KEY_ONES_DIM,  vec![1.0; spec.dim]),
        (KEY_ZEROS_DIM, vec![0.0; spec.dim]),
    ]);

    let x   = vec![0.1_f32; spec.b * spec.s2 * spec.input_dim];
    let cos = vec![1.0_f32; spec.s2 * spec.head_dim / 2];
    let sin = vec![0.0_f32; spec.s2 * spec.head_dim / 2];
    let outs = compiled.run(&[("x", &x), ("freqs_cos", &cos), ("freqs_sin", &sin)]);
    assert_eq!(outs.len(), 1, "expected one output tensor");
    assert_eq!(outs[0].len(), spec.b * spec.s * spec.output_dim,
        "encoder output length mismatch");
}

#[test]
fn decoder_graph_compiles_and_runs() {
    let spec = DecoderSpec {
        b: 1,
        s: 4,
        input_dim:   8,
        encoder_dim: 8,
        dim:         16,
        t_dim:       8,
        n_layers:    2,
        head_dim:    8,
        n_heads:     2,
        hidden_dim:  32,
        norm_eps:    1e-5,
    };

    let graph = build_decoder_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());

    zero_fill_params(&graph, &mut compiled, &[
        (KEY_ONES_DIM,  vec![1.0; spec.dim]),
        (KEY_ZEROS_DIM, vec![0.0; spec.dim]),
        (KEY_TWO_PI,    vec![std::f32::consts::TAU]),
    ]);

    let z       = vec![0.0_f32; spec.b * spec.s * spec.input_dim];
    let enc_out = vec![0.0_f32; spec.b * spec.s * spec.encoder_dim];
    let time_t  = vec![0.5_f32; spec.b];
    let cos     = vec![1.0_f32; spec.s * spec.head_dim / 2];
    let sin     = vec![0.0_f32; spec.s * spec.head_dim / 2];
    let outs = compiled.run(&[
        ("z",         &z),
        ("enc_out",   &enc_out),
        ("time_t",    &time_t),
        ("freqs_cos", &cos),
        ("freqs_sin", &sin),
    ]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), spec.b * spec.s * spec.input_dim,
        "decoder output length mismatch");
}

#[test]
fn try_set_param_rejects_unknown_name() {
    // `Session::compile` populates the param-name set on the
    // CompiledGraph; `try_set_param` should surface typos instead of
    // silently no-oping like `set_param`.
    let spec = EncoderSpec {
        b: 1, s: 4, s2: 8, input_dim: 8, output_dim: 8, dim: 16,
        n_layers: 1, head_dim: 8, n_heads: 2, hidden_dim: 16,
        downsample_factor: 1, norm_eps: 1e-5,
    };
    let mut compiled = rlx::Session::new(rlx::Device::Cpu)
        .compile(build_encoder_graph(&spec));
    let err = compiled.try_set_param("encoder.does_not_exist", &[0.0_f32; 1]);
    assert!(err.is_err(), "unknown param should error");
}

#[test]
fn rope_table_matches_reference_layout() {
    // Layout the loader builds: [max_seqlen, half, 4] with each
    // [pos, h, :] = [cos, -sin, sin, cos]. Spot-check pos=0 yields an
    // identity rotation (cos=1, sin=0).
    let table = build_rope_table(/*head_dim=*/64, /*rope_dim=*/4,
                                  /*max_seqlen=*/50, /*theta=*/10_000.0);
    let half = (64 / 4) / 2;
    for h in 0..half {
        let base = h * 4;
        assert!((table[base]     - 1.0).abs() < 1e-6, "cos@pos0,h={h}");
        assert!((table[base + 1] - 0.0).abs() < 1e-6, "-sin@pos0,h={h}");
        assert!((table[base + 2] - 0.0).abs() < 1e-6, "sin@pos0,h={h}");
        assert!((table[base + 3] - 1.0).abs() < 1e-6, "cos@pos0,h={h}");
    }
}

#[test]
fn precompute_rope_returns_expected_shape() {
    let head_dim = 64;
    let rope_dim = 4;
    let s2 = 4;
    let table = build_rope_table(head_dim, rope_dim, 50, 10_000.0);
    let tok_idx: Vec<i32> = (0..s2 as i32).flat_map(|i| [i, i, i, i]).collect();
    let (cos, sin) = precompute_rope(&tok_idx, &table, head_dim, rope_dim, s2);
    assert_eq!(cos.len(), s2 * head_dim / 2);
    assert_eq!(sin.len(), s2 * head_dim / 2);
}

#[test]
fn preinterleave_doubles_sequence() {
    let registers: Vec<f32> = vec![9.0; 4];
    let tokens = vec![
        1.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
    ];
    let out = preinterleave(&tokens, &registers, 1, 3, 4, 1);
    assert_eq!(out.len(), 6 * 4);
    // Expected: reg, tok0, reg, tok1, reg, tok2
    assert_eq!(&out[0..4],   &[9.0; 4]);
    assert_eq!(&out[4..8],   &[1.0; 4]);
    assert_eq!(&out[8..12],  &[9.0; 4]);
    assert_eq!(&out[12..16], &[2.0; 4]);
    assert_eq!(&out[16..20], &[9.0; 4]);
    assert_eq!(&out[20..24], &[3.0; 4]);
}

#[test]
fn repeat_token_idx_pattern() {
    let tok_idx: Vec<i32> = vec![
        1, 2, 3, 4,
        5, 6, 7, 8,
    ];
    let out = repeat_token_idx(&tok_idx, 2, 1);
    assert_eq!(out, vec![
        1, 2, 3, 4,
        1, 2, 3, 4,
        5, 6, 7, 8,
        5, 6, 7, 8,
    ]);
}
