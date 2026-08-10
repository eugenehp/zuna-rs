//! Safetensors → flat `HashMap<String, Vec<f32>>` loader for the RLX backend.
//!
//! Mirrors the original Burn implementation in `crate::weights` but
//! produces plain `Vec<f32>` buffers (one per parameter) so they can be
//! pushed into an `rlx::CompiledGraph` with `set_param(name, &[f32])`.
//!
//! ## Weight-orientation convention
//!
//! PyTorch / HuggingFace safetensors stores `Linear` weights as `[out, in]`.
//! RLX `g.matmul(input, weight)` expects the rhs to be `[in, out]`. The
//! loader transposes once at load time so the runtime path is straight
//! `mm(x, w)`.
//!
//! The `decoder.t_embedder.weight` Fourier-feature buffer is stored on
//! disk as `[t_dim/2, 1]`; we transpose it to `[1, t_dim/2]` so the graph
//! can do `mm(time_t [B,1,1], weight [1, t_dim/2])`.
//!
//! ## Checkpoint versions
//!
//! Keys are canonicalised with [`canonical_key`] on load, so ZUNA1 and
//! ZUNA1.1 share one namespace. The tensors ZUNA1.1 adds (QK-norms and
//! sandwich norms) are picked up only when [`ModelArch::detect`] finds them.

use std::collections::HashMap;

use half::bf16;
use safetensors::SafeTensors;

use super::graph::{KEY_ONES_DIM, KEY_TWO_PI, KEY_ZEROS_DIM, KEY_ZEROS_HEAD};
use crate::config::{canonical_key, ModelArch, ModelConfig};

/// Parameter buffer with its expected shape.
#[derive(Clone, Debug)]
pub struct ParamBuf {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Map of `name → (data, shape)` containing every parameter the encoder
/// and / or decoder graph expects.
pub type ParamMap = HashMap<String, ParamBuf>;

/// Which half (or both) of the model to load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadScope {
    Encoder,
    Decoder,
    Both,
}

/// Load the entire safetensors file, normalising every tensor to f32 and
/// stripping the leading `model.` prefix.
pub fn load_safetensors(path: &str) -> anyhow::Result<HashMap<String, ParamBuf>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut out = HashMap::with_capacity(st.len());
    for (raw_key, view) in st.tensors() {
        let stripped = raw_key.strip_prefix("model.").unwrap_or(raw_key.as_str());
        let key = canonical_key(stripped).into_owned();
        let shape: Vec<usize> = view.shape().to_vec();
        let data = match view.dtype() {
            safetensors::Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            safetensors::Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => anyhow::bail!("unsupported safetensors dtype {:?} for key {}", other, key),
        };
        out.insert(key, ParamBuf { data, shape });
    }
    Ok(out)
}

/// Detect which ZUNA architecture a loaded (canonicalised) tensor map holds.
pub fn detect_arch(raw: &HashMap<String, ParamBuf>) -> ModelArch {
    ModelArch::detect(|k| raw.contains_key(k))
}

/// Infer the number of attention heads from the wq weight shape.
/// wq is stored as `[n_heads * head_dim, dim]` (PyTorch `[out, in]`).
pub fn infer_n_heads(raw: &HashMap<String, ParamBuf>, head_dim: usize) -> anyhow::Result<usize> {
    anyhow::ensure!(head_dim > 0, "head_dim must be > 0");
    let key = "encoder.layers.0.attention.wq.weight";
    let p = raw
        .get(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))?;
    anyhow::ensure!(
        p.shape.len() >= 2,
        "wq weight must be 2-D, got shape {:?}",
        p.shape
    );
    Ok(p.shape[0] / head_dim)
}

/// Transpose a `[rows, cols]` row-major matrix in place into a new buffer.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Pull a tensor from `raw` and transpose it from `[out_dim, in_dim]` to
/// `[in_dim, out_dim]`. Used for every `Linear.weight`.
fn take_linear_w(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    let p = raw
        .remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))?;
    anyhow::ensure!(
        p.shape.len() == 2,
        "Linear weight {key} must be 2-D, got {:?}",
        p.shape
    );
    let (out_d, in_d) = (p.shape[0], p.shape[1]);
    let data = transpose(&p.data, out_d, in_d);
    Ok(ParamBuf {
        data,
        shape: vec![in_d, out_d],
    })
}

fn take(raw: &mut HashMap<String, ParamBuf>, key: &str) -> anyhow::Result<ParamBuf> {
    raw.remove(key)
        .ok_or_else(|| anyhow::anyhow!("missing weight key: {key}"))
}

/// Build the full encoder parameter map (graph-ready) from a raw safetensors
/// map and the model config. Also returns the inferred `n_heads` and the
/// detected [`ModelArch`].
pub fn build_encoder_params(
    raw: &mut HashMap<String, ParamBuf>,
    cfg: &ModelConfig,
) -> anyhow::Result<(ParamMap, usize, ModelArch)> {
    let n_heads = infer_n_heads(raw, cfg.head_dim)?;
    let arch = detect_arch(raw);
    let mut p = ParamMap::new();

    p.insert(
        "encoder.tok_embeddings.weight".into(),
        take_linear_w(raw, "encoder.tok_embeddings.weight")?,
    );
    p.insert(
        "encoder.tok_embeddings.bias".into(),
        take(raw, "encoder.tok_embeddings.bias")?,
    );

    // registers: [1, input_dim] on disk; we use it as [input_dim] (squeeze).
    let regs = take(raw, "encoder.registers")?;
    anyhow::ensure!(
        regs.shape == vec![1, cfg.input_dim],
        "encoder.registers shape mismatch: {:?}",
        regs.shape
    );
    p.insert(
        "encoder.registers".into(),
        ParamBuf {
            data: regs.data,
            shape: vec![cfg.input_dim],
        },
    );

    p.insert(
        "encoder.norm.weight".into(),
        take(raw, "encoder.norm.weight")?,
    );

    p.insert(
        "encoder.output.weight".into(),
        take_linear_w(raw, "encoder.output.weight")?,
    );

    for i in 0..cfg.n_layers {
        let q = format!("encoder.layers.{i}");
        p.insert(
            format!("{q}.attention_norm.weight"),
            take(raw, &format!("{q}.attention_norm.weight"))?,
        );
        for which in ["wq", "wk", "wv", "wo"] {
            p.insert(
                format!("{q}.attention.{which}.weight"),
                take_linear_w(raw, &format!("{q}.attention.{which}.weight"))?,
            );
        }
        take_qk_norms(raw, &mut p, arch, &format!("{q}.attention"))?;
        p.insert(
            format!("{q}.ffn_norm.weight"),
            take(raw, &format!("{q}.ffn_norm.weight"))?,
        );
        for which in ["w1", "w2", "w3"] {
            p.insert(
                format!("{q}.feed_forward.{which}.weight"),
                take_linear_w(raw, &format!("{q}.feed_forward.{which}.weight"))?,
            );
        }
        take_post_norms(
            raw,
            &mut p,
            arch,
            &q,
            &["attention_norm_post", "ffn_norm_post"],
        )?;
    }

    insert_aux_params(&mut p, cfg.dim, cfg.head_dim, arch, false);
    Ok((p, n_heads, arch))
}

/// Build the full decoder parameter map (graph-ready).
/// Returns `(map, n_heads, arch)`.
pub fn build_decoder_params(
    raw: &mut HashMap<String, ParamBuf>,
    cfg: &ModelConfig,
) -> anyhow::Result<(ParamMap, usize, ModelArch)> {
    let n_heads = infer_n_heads(raw, cfg.head_dim)?;
    let arch = detect_arch(raw);
    let mut p = ParamMap::new();

    p.insert(
        "decoder.tok_embeddings.weight".into(),
        take_linear_w(raw, "decoder.tok_embeddings.weight")?,
    );
    p.insert(
        "decoder.tok_embeddings.bias".into(),
        take(raw, "decoder.tok_embeddings.bias")?,
    );

    // FourierConditioner: stored on disk as [t_dim/2, 1]; transpose to
    // [1, t_dim/2] so the graph can mm(time_t, weight) directly.
    let tw = take(raw, "decoder.t_embedder.weight")?;
    anyhow::ensure!(
        tw.shape == vec![cfg.t_dim / 2, 1],
        "decoder.t_embedder.weight shape mismatch: {:?}",
        tw.shape
    );
    let tw_t = transpose(&tw.data, cfg.t_dim / 2, 1);
    p.insert(
        "decoder.t_embedder.weight".into(),
        ParamBuf {
            data: tw_t,
            shape: vec![1, cfg.t_dim / 2],
        },
    );

    p.insert(
        "decoder.t_embedder.proj.weight".into(),
        take_linear_w(raw, "decoder.t_embedder.proj.weight")?,
    );
    p.insert(
        "decoder.t_embedder.proj.bias".into(),
        take(raw, "decoder.t_embedder.proj.bias")?,
    );

    p.insert(
        "decoder.encoder_proj.weight".into(),
        take_linear_w(raw, "decoder.encoder_proj.weight")?,
    );
    p.insert(
        "decoder.encoder_proj.bias".into(),
        take(raw, "decoder.encoder_proj.bias")?,
    );

    p.insert(
        "decoder.norm.weight.weight".into(),
        take_linear_w(raw, "decoder.norm.weight.weight")?,
    );
    p.insert(
        "decoder.norm.weight.bias".into(),
        take(raw, "decoder.norm.weight.bias")?,
    );

    p.insert(
        "decoder.output.weight".into(),
        take_linear_w(raw, "decoder.output.weight")?,
    );

    for i in 0..cfg.n_layers {
        let q = format!("decoder.layers.{i}");
        for ada in [
            "cross_attention_x_norm",
            "cross_attention_y_norm",
            "attention_norm",
            "ffn_norm",
        ] {
            p.insert(
                format!("{q}.{ada}.weight.weight"),
                take_linear_w(raw, &format!("{q}.{ada}.weight.weight"))?,
            );
            p.insert(
                format!("{q}.{ada}.weight.bias"),
                take(raw, &format!("{q}.{ada}.weight.bias"))?,
            );
        }
        for attn in ["cross_attention", "attention"] {
            for which in ["wq", "wk", "wv", "wo"] {
                p.insert(
                    format!("{q}.{attn}.{which}.weight"),
                    take_linear_w(raw, &format!("{q}.{attn}.{which}.weight"))?,
                );
            }
            take_qk_norms(raw, &mut p, arch, &format!("{q}.{attn}"))?;
        }
        for which in ["w1", "w2", "w3"] {
            p.insert(
                format!("{q}.feed_forward.{which}.weight"),
                take_linear_w(raw, &format!("{q}.feed_forward.{which}.weight"))?,
            );
        }
        take_post_norms(
            raw,
            &mut p,
            arch,
            &q,
            &[
                "cross_attention_norm_post",
                "attention_norm_post",
                "ffn_norm_post",
            ],
        )?;
    }

    insert_aux_params(&mut p, cfg.dim, cfg.head_dim, arch, true);
    Ok((p, n_heads, arch))
}

/// Move the ZUNA1.1 `q_norm` / `k_norm` scales for one attention module.
/// No-op on checkpoints without QK-norm.
fn take_qk_norms(
    raw: &mut HashMap<String, ParamBuf>,
    p: &mut ParamMap,
    arch: ModelArch,
    attn_prefix: &str,
) -> anyhow::Result<()> {
    if !arch.qk_norm {
        return Ok(());
    }
    for which in ["q_norm", "k_norm"] {
        let key = format!("{attn_prefix}.{which}.weight");
        let buf = take(raw, &key)?;
        p.insert(key, buf);
    }
    Ok(())
}

/// Move the ZUNA1.1 sandwich-norm scales for one transformer block.
/// No-op on checkpoints without sandwich norms.
fn take_post_norms(
    raw: &mut HashMap<String, ParamBuf>,
    p: &mut ParamMap,
    arch: ModelArch,
    block_prefix: &str,
    names: &[&str],
) -> anyhow::Result<()> {
    if !arch.sandwich_norm {
        return Ok(());
    }
    for name in names {
        let key = format!("{block_prefix}.{name}.weight");
        let buf = take(raw, &key)?;
        p.insert(key, buf);
    }
    Ok(())
}

/// Insert the auxiliary constants the graph needs but the safetensors
/// file doesn't supply. `with_two_pi` is `true` for the decoder (Fourier
/// conditioner uses `2π`).
fn insert_aux_params(
    p: &mut ParamMap,
    dim: usize,
    head_dim: usize,
    arch: ModelArch,
    with_two_pi: bool,
) {
    p.insert(
        KEY_ZEROS_DIM.into(),
        ParamBuf {
            data: vec![0.0; dim],
            shape: vec![dim],
        },
    );
    p.insert(
        KEY_ONES_DIM.into(),
        ParamBuf {
            data: vec![1.0; dim],
            shape: vec![dim],
        },
    );
    if arch.qk_norm {
        p.insert(
            KEY_ZEROS_HEAD.into(),
            ParamBuf {
                data: vec![0.0; head_dim],
                shape: vec![head_dim],
            },
        );
    }
    if with_two_pi {
        p.insert(
            KEY_TWO_PI.into(),
            ParamBuf {
                data: vec![std::f32::consts::TAU],
                shape: vec![1],
            },
        );
    }
}

/// Push every parameter in `params` into the given compiled graph.
///
/// Skips entries whose name is not declared as a parameter in the graph —
/// this lets the loader hold params that only one of the encoder or
/// decoder graph references (e.g. `registers` is only consumed by the
/// CPU-side interleaver, never by the graph).
pub fn apply_params(compiled: &mut rlx::CompiledGraph, params: &ParamMap) {
    for (name, buf) in params {
        // Best-effort set: some param names (registers, KEY_TWO_PI for the
        // encoder graph) aren't declared in every graph. The runtime is
        // expected to silently ignore unknown names; if it errors, the
        // caller can pre-filter via `params.contains_key`.
        compiled.set_param(name, &buf.data);
    }
}

/// Convenience: load + split safetensors into (encoder, decoder) maps.
pub fn load_split(
    weights_path: &str,
    cfg: &ModelConfig,
    scope: LoadScope,
) -> anyhow::Result<(Option<ParamMap>, Option<ParamMap>, usize, ModelArch)> {
    let mut raw = load_safetensors(weights_path)?;
    let n_heads = infer_n_heads(&raw, cfg.head_dim)?;
    let arch = detect_arch(&raw);
    let enc = if matches!(scope, LoadScope::Encoder | LoadScope::Both) {
        // Clone raw on demand so we can build both maps.
        let mut r = raw.clone();
        Some(build_encoder_params(&mut r, cfg)?.0)
    } else {
        None
    };
    let dec = if matches!(scope, LoadScope::Decoder | LoadScope::Both) {
        Some(build_decoder_params(&mut raw, cfg)?.0)
    } else {
        None
    };
    Ok((enc, dec, n_heads, arch))
}
