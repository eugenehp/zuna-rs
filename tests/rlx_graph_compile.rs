//! IR-level tests for the RLX encoder / decoder graphs.
//!
//! No safetensors weights required: build the graph, compile it,
//! push zero-filled buffers as every parameter, and assert the runtime
//! produces an output of the expected shape. Catches op-name and
//! shape-builder typos that the parity tests can only surface when
//! real weights are available.

use zuna_rs::rlx::graph::{
    build_decoder_graph, build_encoder_graph, DecoderSpec, EncoderSpec, KEY_ONES_DIM, KEY_TWO_PI,
    KEY_ZEROS_DIM, KEY_ZEROS_HEAD,
};
use zuna_rs::rlx::rope_helpers::{
    build_rope_table, precompute_rope, preinterleave, repeat_token_idx,
};
use zuna_rs::ModelArch;

/// Push a buffer for every `Op::Param` declared in `graph`. The
/// `overrides` map provides specific values for named params
/// (e.g. the `__zuna.ones_dim` / `__zuna.zeros_dim` constants);
/// everything else gets a zero vector of the declared length.
fn zero_fill_params(
    graph: &rlx::Graph,
    compiled: &mut rlx::CompiledGraph,
    overrides: &[(&str, Vec<f32>)],
) {
    use rlx::Op;
    let mut ovr: std::collections::HashMap<&str, Vec<f32>> =
        overrides.iter().map(|(k, v)| (*k, v.clone())).collect();
    for node in graph.nodes() {
        let Op::Param { name } = &node.op else {
            continue;
        };
        let n = node
            .shape
            .num_elements()
            .expect("param shape must be static");
        let buf = ovr.remove(name.as_str()).unwrap_or_else(|| vec![0.0; n]);
        assert_eq!(
            buf.len(),
            n,
            "param {name}: buffer len {} != shape {}",
            buf.len(),
            n
        );
        compiled.set_param(name, &buf);
    }
}

/// Tiny synthetic shape — proves the IR builders are valid; not
/// intended to produce a meaningful EEG embedding.
fn encoder_spec(arch: ModelArch) -> EncoderSpec {
    EncoderSpec {
        b: 1,
        s: 4,
        s2: 8,
        input_dim: 8,
        output_dim: 8,
        dim: 16,
        n_layers: 2,
        head_dim: 8,
        n_heads: 2,
        hidden_dim: 32,
        downsample_factor: 1,
        norm_eps: 1e-5,
        arch,
    }
}

fn decoder_spec(arch: ModelArch) -> DecoderSpec {
    DecoderSpec {
        b: 1,
        s: 4,
        input_dim: 8,
        encoder_dim: 8,
        dim: 16,
        t_dim: 8,
        n_layers: 2,
        head_dim: 8,
        n_heads: 2,
        hidden_dim: 32,
        norm_eps: 1e-5,
        arch,
    }
}

fn run_encoder_graph(spec: &EncoderSpec) -> Vec<Vec<f32>> {
    let graph = build_encoder_graph(spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());

    zero_fill_params(
        &graph,
        &mut compiled,
        &[
            (KEY_ONES_DIM, vec![1.0; spec.dim]),
            (KEY_ZEROS_DIM, vec![0.0; spec.dim]),
            (KEY_ZEROS_HEAD, vec![0.0; spec.head_dim]),
        ],
    );

    let x = vec![0.1_f32; spec.b * spec.s2 * spec.input_dim];
    let cos = vec![1.0_f32; spec.s2 * spec.head_dim / 2];
    let sin = vec![0.0_f32; spec.s2 * spec.head_dim / 2];
    compiled.run(&[("x", &x), ("freqs_cos", &cos), ("freqs_sin", &sin)])
}

fn run_decoder_graph(spec: &DecoderSpec) -> Vec<Vec<f32>> {
    let graph = build_decoder_graph(spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());

    zero_fill_params(
        &graph,
        &mut compiled,
        &[
            (KEY_ONES_DIM, vec![1.0; spec.dim]),
            (KEY_ZEROS_DIM, vec![0.0; spec.dim]),
            (KEY_ZEROS_HEAD, vec![0.0; spec.head_dim]),
            (KEY_TWO_PI, vec![std::f32::consts::TAU]),
        ],
    );

    let z = vec![0.0_f32; spec.b * spec.s * spec.input_dim];
    let enc_out = vec![0.0_f32; spec.b * spec.s * spec.encoder_dim];
    let time_t = vec![0.5_f32; spec.b];
    let cos = vec![1.0_f32; spec.s * spec.head_dim / 2];
    let sin = vec![0.0_f32; spec.s * spec.head_dim / 2];
    compiled.run(&[
        ("z", &z),
        ("enc_out", &enc_out),
        ("time_t", &time_t),
        ("freqs_cos", &cos),
        ("freqs_sin", &sin),
    ])
}

/// Names of every `Op::Param` in a graph.
fn param_names(graph: &rlx::Graph) -> std::collections::HashSet<String> {
    use rlx::Op;
    graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn encoder_graph_compiles_and_runs() {
    for arch in [ModelArch::ZUNA1, ModelArch::ZUNA1_1] {
        let spec = encoder_spec(arch);
        let outs = run_encoder_graph(&spec);
        assert_eq!(
            outs.len(),
            1,
            "{}: expected one output tensor",
            arch.label()
        );
        assert_eq!(
            outs[0].len(),
            spec.b * spec.s * spec.output_dim,
            "{}: encoder output length mismatch",
            arch.label()
        );
    }
}

#[test]
fn decoder_graph_compiles_and_runs() {
    for arch in [ModelArch::ZUNA1, ModelArch::ZUNA1_1] {
        let spec = decoder_spec(arch);
        let outs = run_decoder_graph(&spec);
        assert_eq!(
            outs.len(),
            1,
            "{}: expected one output tensor",
            arch.label()
        );
        assert_eq!(
            outs[0].len(),
            spec.b * spec.s * spec.input_dim,
            "{}: decoder output length mismatch",
            arch.label()
        );
    }
}

/// ZUNA1.1 declares QK-norm and sandwich-norm params; ZUNA1 declares neither.
/// Keeps the graph builders and the [`zuna_rs::rlx::weights`] loader — which
/// only *supplies* those params when the same flags are set — in lock step.
#[test]
fn zuna11_graphs_declare_the_extra_norms() {
    let enc_11 = param_names(&build_encoder_graph(&encoder_spec(ModelArch::ZUNA1_1)));
    let enc_1 = param_names(&build_encoder_graph(&encoder_spec(ModelArch::ZUNA1)));
    for key in [
        "encoder.layers.0.attention.q_norm.weight",
        "encoder.layers.0.attention.k_norm.weight",
        "encoder.layers.0.attention_norm_post.weight",
        "encoder.layers.0.ffn_norm_post.weight",
        KEY_ZEROS_HEAD,
    ] {
        assert!(enc_11.contains(key), "ZUNA1.1 encoder graph missing {key}");
        assert!(
            !enc_1.contains(key),
            "ZUNA1 encoder graph should not declare {key}"
        );
    }
    // Shared keys stay spelled the ZUNA1 way on both.
    for key in [
        "encoder.layers.0.attention_norm.weight",
        "encoder.norm.weight",
    ] {
        assert!(
            enc_11.contains(key) && enc_1.contains(key),
            "{key} must be version-neutral"
        );
    }

    let dec_11 = param_names(&build_decoder_graph(&decoder_spec(ModelArch::ZUNA1_1)));
    let dec_1 = param_names(&build_decoder_graph(&decoder_spec(ModelArch::ZUNA1)));
    for key in [
        "decoder.layers.0.attention.q_norm.weight",
        "decoder.layers.0.cross_attention.k_norm.weight",
        "decoder.layers.0.cross_attention_norm_post.weight",
        "decoder.layers.0.attention_norm_post.weight",
        "decoder.layers.0.ffn_norm_post.weight",
    ] {
        assert!(dec_11.contains(key), "ZUNA1.1 decoder graph missing {key}");
        assert!(
            !dec_1.contains(key),
            "ZUNA1 decoder graph should not declare {key}"
        );
    }
}

/// The ZUNA1.1 QK-norm feeds a `[B,S,H,Dh]` tensor to `rms_norm` with a
/// `[Dh]` gamma, expecting each `(b,s,h)` row to be normalised independently —
/// exactly `F.rms_norm(x, (head_dim,), weight, eps)` on the PyTorch side.
/// This pins that contract down numerically.
#[test]
fn rms_norm_normalises_each_head_row() {
    use rlx::ir::GraphExt;
    use rlx::prelude::*;

    const B: usize = 1;
    const S: usize = 2;
    const H: usize = 2;
    const DH: usize = 4;
    const EPS: f32 = 1e-5;

    let mut g = Graph::new("qk_norm_probe");
    let x = g.input("x", Shape::new(&[B, S, H, DH], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[DH], DType::F32));
    let beta = g.param("beta", Shape::new(&[DH], DType::F32));
    let out = g.rms_norm(x, gamma, beta, EPS);
    g.set_outputs(vec![out]);

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(g);
    // Gamma varies per channel so a transposed / mis-broadcast gamma would show.
    let gamma_v = [1.0_f32, 2.0, 3.0, 4.0];
    compiled.set_param("gamma", &gamma_v);
    compiled.set_param("beta", &[0.0_f32; DH]);

    // Each row has a deliberately different scale.
    let rows: [[f32; DH]; B * S * H] = [
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
        [-1.0, 1.0, -1.0, 1.0],
        [0.5, 0.0, -0.5, 0.25],
    ];
    let input: Vec<f32> = rows.iter().flatten().copied().collect();
    let got = compiled.run(&[("x", &input)]).remove(0);
    assert_eq!(got.len(), B * S * H * DH);

    for (r, row) in rows.iter().enumerate() {
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / DH as f32;
        let inv_rms = (mean_sq + EPS).sqrt().recip();
        for c in 0..DH {
            let want = row[c] * inv_rms * gamma_v[c];
            assert!(
                (got[r * DH + c] - want).abs() < 1e-5,
                "row {r} col {c}: got {}, want {want}",
                got[r * DH + c]
            );
        }
    }
    // Rows 0 and 1 differ only by a factor of 10 — RMS-normalising each row
    // independently must map them onto the same output.
    for c in 0..DH {
        assert!(
            (got[c] - got[DH + c]).abs() < 1e-5,
            "scale-invariance broken at col {c}: {} vs {}",
            got[c],
            got[DH + c]
        );
    }
}

#[test]
fn rope_table_matches_reference_layout() {
    // Layout the loader builds: [max_seqlen, half, 4] with each
    // [pos, h, :] = [cos, -sin, sin, cos]. Spot-check pos=0 yields an
    // identity rotation (cos=1, sin=0).
    let table = build_rope_table(
        /*head_dim=*/ 64, /*rope_dim=*/ 4, /*max_seqlen=*/ 50,
        /*theta=*/ 10_000.0,
    );
    let half = (64 / 4) / 2;
    for h in 0..half {
        let base = h * 4;
        assert!((table[base] - 1.0).abs() < 1e-6, "cos@pos0,h={h}");
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
    let tokens = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let out = preinterleave(&tokens, &registers, 1, 3, 4, 1);
    assert_eq!(out.len(), 6 * 4);
    // Expected: reg, tok0, reg, tok1, reg, tok2
    assert_eq!(&out[0..4], &[9.0; 4]);
    assert_eq!(&out[4..8], &[1.0; 4]);
    assert_eq!(&out[8..12], &[9.0; 4]);
    assert_eq!(&out[12..16], &[2.0; 4]);
    assert_eq!(&out[16..20], &[9.0; 4]);
    assert_eq!(&out[20..24], &[3.0; 4]);
}

#[test]
fn repeat_token_idx_pattern() {
    let tok_idx: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let out = repeat_token_idx(&tok_idx, 2, 1);
    assert_eq!(out, vec![1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8,]);
}
