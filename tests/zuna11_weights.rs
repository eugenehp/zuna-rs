//! Synthetic-checkpoint round trip for the ZUNA1 / ZUNA1.1 weight loaders.
//!
//! Writes a miniature safetensors file that uses each checkpoint's exact key
//! spelling — including ZUNA1.1's extra `norm.` nesting, QK-norms and sandwich
//! norms — then asserts two invariants:
//!
//! 1. **Nothing is dropped.** Every tensor in the file is consumed by the
//!    param builder, so a ZUNA1.1-only weight can't be silently ignored.
//! 2. **Nothing is missing.** Every `Op::Param` the compiled graph declares is
//!    supplied by the loader, with a matching element count.
//!
//! Together these keep `rlx::graph` and `rlx::weights` from drifting apart.

use std::collections::HashMap;

use safetensors::tensor::TensorView;
use safetensors::Dtype;

use zuna_rs::rlx::graph::{build_decoder_graph, build_encoder_graph, DecoderSpec, EncoderSpec};
use zuna_rs::rlx::weights::{
    build_decoder_params, build_encoder_params, detect_arch, load_safetensors, ParamMap,
};
use zuna_rs::{ModelArch, ModelConfig};

// ── Miniature architecture ───────────────────────────────────────────────────

const DIM: usize = 16;
const N_LAYERS: usize = 2;
const HEAD_DIM: usize = 8;
const N_HEADS: usize = 2; // inferred by the loader from wq: [16, 16] / 8
const INPUT_DIM: usize = 8;
const OUT_DIM: usize = 8;
const T_DIM: usize = 8;
const HIDDEN: usize = 48; // multiple_of=8 → 8 * ceil((2*4*16/3) / 8)

fn config() -> ModelConfig {
    let cfg: ModelConfig = serde_json::from_str(&format!(
        r#"{{
        "dim": {DIM}, "n_layers": {N_LAYERS}, "head_dim": {HEAD_DIM},
        "input_dim": {INPUT_DIM}, "encoder_output_dim": {OUT_DIM},
        "encoder_latent_downsample_factor": 1,
        "t_dim": {T_DIM}, "multiple_of": 8,
        "max_seqlen": 16, "rope_dim": 4, "rope_theta": 10000.0,
        "stft_global_sigma": 0.1
    }}"#
    ))
    .expect("test config parses");
    assert_eq!(cfg.ffn_hidden_dim(), HIDDEN);
    cfg
}

// ── Checkpoint synthesis ─────────────────────────────────────────────────────

/// Name of a plain RMSNorm scale as the given checkpoint spells it: ZUNA1
/// stores it on the module, ZUNA1.1 on the wrapped `torch.nn.RMSNorm`.
fn rms(arch: ModelArch, module: &str) -> String {
    if arch == ModelArch::ZUNA1_1 {
        format!("{module}.norm.weight")
    } else {
        format!("{module}.weight")
    }
}

/// `(key, shape)` for every tensor a checkpoint of this architecture holds.
/// Linear weights use PyTorch's `[out, in]` order, as on disk.
fn tensor_spec(arch: ModelArch) -> Vec<(String, Vec<usize>)> {
    let mut t: Vec<(String, Vec<usize>)> = vec![
        // Encoder stem
        (
            "model.encoder.tok_embeddings.weight".into(),
            vec![DIM, INPUT_DIM],
        ),
        ("model.encoder.tok_embeddings.bias".into(), vec![DIM]),
        ("model.encoder.registers".into(), vec![1, INPUT_DIM]),
        (format!("model.{}", rms(arch, "encoder.norm")), vec![DIM]),
        ("model.encoder.output.weight".into(), vec![OUT_DIM, DIM]),
        // Decoder stem
        (
            "model.decoder.tok_embeddings.weight".into(),
            vec![DIM, INPUT_DIM],
        ),
        ("model.decoder.tok_embeddings.bias".into(), vec![DIM]),
        ("model.decoder.t_embedder.weight".into(), vec![T_DIM / 2, 1]),
        (
            "model.decoder.t_embedder.proj.weight".into(),
            vec![T_DIM, T_DIM],
        ),
        ("model.decoder.t_embedder.proj.bias".into(), vec![T_DIM]),
        (
            "model.decoder.encoder_proj.weight".into(),
            vec![DIM, OUT_DIM],
        ),
        ("model.decoder.encoder_proj.bias".into(), vec![DIM]),
        ("model.decoder.norm.weight.weight".into(), vec![DIM, T_DIM]),
        ("model.decoder.norm.weight.bias".into(), vec![DIM]),
        ("model.decoder.output.weight".into(), vec![INPUT_DIM, DIM]),
    ];

    let qkv = DIM; // n_heads * head_dim
    for i in 0..N_LAYERS {
        // ── encoder block ──
        let p = format!("model.encoder.layers.{i}");
        t.push((format!("{p}.{}", rms(arch, "attention_norm")), vec![DIM]));
        for w in ["wq", "wk", "wv", "wo"] {
            t.push((format!("{p}.attention.{w}.weight"), vec![qkv, DIM]));
        }
        t.push((format!("{p}.{}", rms(arch, "ffn_norm")), vec![DIM]));
        for (w, shape) in [
            ("w1", vec![HIDDEN, DIM]),
            ("w2", vec![DIM, HIDDEN]),
            ("w3", vec![HIDDEN, DIM]),
        ] {
            t.push((format!("{p}.feed_forward.{w}.weight"), shape));
        }
        if arch.qk_norm {
            for n in ["q_norm", "k_norm"] {
                t.push((format!("{p}.attention.{n}.norm.weight"), vec![HEAD_DIM]));
            }
        }
        if arch.sandwich_norm {
            for n in ["attention_norm_post", "ffn_norm_post"] {
                t.push((format!("{p}.{n}.norm.weight"), vec![DIM]));
            }
        }

        // ── decoder block ──
        let p = format!("model.decoder.layers.{i}");
        for ada in [
            "cross_attention_x_norm",
            "cross_attention_y_norm",
            "attention_norm",
            "ffn_norm",
        ] {
            t.push((format!("{p}.{ada}.weight.weight"), vec![DIM, T_DIM]));
            t.push((format!("{p}.{ada}.weight.bias"), vec![DIM]));
        }
        for attn in ["cross_attention", "attention"] {
            for w in ["wq", "wk", "wv", "wo"] {
                t.push((format!("{p}.{attn}.{w}.weight"), vec![qkv, DIM]));
            }
            if arch.qk_norm {
                for n in ["q_norm", "k_norm"] {
                    t.push((format!("{p}.{attn}.{n}.norm.weight"), vec![HEAD_DIM]));
                }
            }
        }
        for (w, shape) in [
            ("w1", vec![HIDDEN, DIM]),
            ("w2", vec![DIM, HIDDEN]),
            ("w3", vec![HIDDEN, DIM]),
        ] {
            t.push((format!("{p}.feed_forward.{w}.weight"), shape));
        }
        if arch.sandwich_norm {
            for n in [
                "cross_attention_norm_post",
                "attention_norm_post",
                "ffn_norm_post",
            ] {
                t.push((format!("{p}.{n}.norm.weight"), vec![DIM]));
            }
        }
    }
    t
}

/// Serialize a zero-filled checkpoint with the given key/shape spec.
fn write_checkpoint(arch: ModelArch, path: &std::path::Path) {
    let spec = tensor_spec(arch);
    let buffers: Vec<Vec<u8>> = spec
        .iter()
        .map(|(_, shape)| vec![0u8; shape.iter().product::<usize>() * 4])
        .collect();
    let views: Vec<(String, TensorView)> = spec
        .iter()
        .zip(&buffers)
        .map(|((name, shape), buf)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), buf).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, path).expect("write checkpoint");
}

// ── Assertions ───────────────────────────────────────────────────────────────

/// Every `Op::Param` the graph declares must be present in `params` with a
/// matching element count.
fn assert_graph_params_supplied(graph: &rlx::Graph, params: &ParamMap, what: &str) {
    use rlx::Op;
    let mut checked = 0;
    for node in graph.nodes() {
        let Op::Param { name } = &node.op else {
            continue;
        };
        let want = node
            .shape
            .num_elements()
            .expect("param shape must be static");
        let buf = params
            .get(name.as_str())
            .unwrap_or_else(|| panic!("{what}: graph declares {name}, loader supplied nothing"));
        assert_eq!(
            buf.data.len(),
            want,
            "{what}: {name} has {} elements, graph wants {want}",
            buf.data.len()
        );
        checked += 1;
    }
    assert!(checked > 0, "{what}: graph declared no params");
}

fn encoder_spec(cfg: &ModelConfig, arch: ModelArch, n_heads: usize) -> EncoderSpec {
    EncoderSpec {
        b: 1,
        s: 4,
        s2: 8,
        input_dim: cfg.input_dim,
        output_dim: cfg.encoder_output_dim,
        dim: cfg.dim,
        n_layers: cfg.n_layers,
        head_dim: cfg.head_dim,
        n_heads,
        hidden_dim: cfg.ffn_hidden_dim(),
        downsample_factor: cfg.encoder_latent_downsample_factor,
        norm_eps: cfg.norm_eps as f32,
        arch,
    }
}

fn decoder_spec(cfg: &ModelConfig, arch: ModelArch, n_heads: usize) -> DecoderSpec {
    DecoderSpec {
        b: 1,
        s: 4,
        input_dim: cfg.input_dim,
        encoder_dim: cfg.encoder_output_dim,
        dim: cfg.dim,
        t_dim: cfg.t_dim,
        n_layers: cfg.n_layers,
        head_dim: cfg.head_dim,
        n_heads,
        hidden_dim: cfg.ffn_hidden_dim(),
        norm_eps: cfg.norm_eps as f32,
        arch,
    }
}

/// Keys left in `raw` under `prefix` after the param builder has run.
fn leftovers(raw: &HashMap<String, impl Sized>, prefix: &str) -> Vec<String> {
    let mut v: Vec<String> = raw
        .keys()
        .filter(|k| k.starts_with(prefix))
        .cloned()
        .collect();
    v.sort();
    v
}

fn round_trip(arch: ModelArch) {
    let cfg = config();
    let dir = std::env::temp_dir().join(format!("zuna_rs_test_{}", arch.label()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("model.safetensors");
    write_checkpoint(arch, &path);

    let raw = load_safetensors(path.to_str().unwrap()).expect("load checkpoint");
    assert_eq!(
        detect_arch(&raw),
        arch,
        "{}: architecture misdetected from the weight names",
        arch.label()
    );

    let mut enc_raw = raw.clone();
    let (enc_params, n_heads, enc_arch) =
        build_encoder_params(&mut enc_raw, &cfg).expect("encoder params");
    assert_eq!(n_heads, N_HEADS);
    assert_eq!(enc_arch, arch);
    assert_eq!(
        leftovers(&enc_raw, "encoder."),
        Vec::<String>::new(),
        "{}: encoder tensors left unconsumed",
        arch.label()
    );
    assert_graph_params_supplied(
        &build_encoder_graph(&encoder_spec(&cfg, arch, n_heads)),
        &enc_params,
        &format!("{} encoder", arch.label()),
    );

    let mut dec_raw = raw;
    let (dec_params, _, dec_arch) =
        build_decoder_params(&mut dec_raw, &cfg).expect("decoder params");
    assert_eq!(dec_arch, arch);
    assert_eq!(
        leftovers(&dec_raw, "decoder."),
        Vec::<String>::new(),
        "{}: decoder tensors left unconsumed",
        arch.label()
    );
    assert_graph_params_supplied(
        &build_decoder_graph(&decoder_spec(&cfg, arch, n_heads)),
        &dec_params,
        &format!("{} decoder", arch.label()),
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn zuna11_checkpoint_round_trips() {
    round_trip(ModelArch::ZUNA1_1);
}

/// The ZUNA1 checkpoint must keep loading unchanged.
#[test]
fn zuna1_checkpoint_still_round_trips() {
    round_trip(ModelArch::ZUNA1);
}
