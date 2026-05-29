//! Parity test: RLX-backed `ZunaEncoder` vs the Python (NumPy) reference.
//!
//! Requires a `data/python_reference.safetensors` produced by
//! `scripts/dump_python_reference.py` (see that script's docstring for
//! how to generate it). The reference file holds, for a single epoch:
//!
//! ```text
//! encoder_input  [S, 32]   float32 — same buffer Python ingested
//! tok_idx        [S, 4]    int32   — same indices Python used
//! embedding      [S, 32]   float32 — Python NumPy encoder output
//! ```
//!
//! The test loads `(encoder_input, tok_idx)`, feeds them into the RLX
//! encoder, and asserts the output matches the Python `embedding` within
//! tolerance. We also report cosine similarity since the user asked for
//! it explicitly.
//!
//! ```text
//! cargo test --release --test parity_rlx_vs_python -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use zuna_rs::rlx::ZunaEncoder;

/// Pick the RLX device based on the `ZUNA_RLX_DEVICE` env var.
fn pick_rlx_device() -> rlx::Device {
    match std::env::var("ZUNA_RLX_DEVICE").as_deref().unwrap_or("cpu") {
        "cpu"   => rlx::Device::Cpu,
        "metal" => rlx::Device::Metal,
        "mlx"   => rlx::Device::Mlx,
        other   => panic!("unknown ZUNA_RLX_DEVICE: {other:?}"),
    }
}

fn locate_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights = std::env::var("ZUNA_WEIGHTS").ok().map(PathBuf::from)
        .or_else(|| {
            let p = manifest.join("data/model.safetensors");
            p.exists().then_some(p)
        })?;
    let config = std::env::var("ZUNA_CONFIG").ok().map(PathBuf::from)
        .or_else(|| {
            let p = manifest.join("data/config.json");
            p.exists().then_some(p)
        })?;
    let py_ref = std::env::var("ZUNA_PYTHON_REF").ok().map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("data/python_reference.safetensors"));
    if !py_ref.exists() { return None; }
    Some((weights, config, py_ref))
}

fn read_f32_tensor(st: &SafeTensors, key: &str) -> Vec<f32> {
    let v = st.tensor(key).unwrap_or_else(|_| panic!("missing key {key}"));
    assert_eq!(v.dtype(), safetensors::Dtype::F32, "{key} must be f32");
    v.data().chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_i32_tensor(st: &SafeTensors, key: &str) -> Vec<i32> {
    let v = st.tensor(key).unwrap_or_else(|_| panic!("missing key {key}"));
    match v.dtype() {
        safetensors::Dtype::I32 => v.data().chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        safetensors::Dtype::I64 => v.data().chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as i32)
            .collect(),
        other => panic!("{key}: unsupported int dtype {other:?}"),
    }
}

fn shape_of(st: &SafeTensors, key: &str) -> Vec<usize> {
    st.tensor(key).expect(key).shape().to_vec()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0_f64;
    let mut na  = 0.0_f64;
    let mut nb  = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na  += x * x;
        nb  += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
fn rlx_encoder_matches_python_reference() {
    let (weights_path, config_path, ref_path) = match locate_paths() {
        Some(t) => t,
        None => {
            eprintln!("\n[SKIP] python parity test — missing inputs.");
            eprintln!("  Need:");
            eprintln!("    * ZUNA weights at data/model.safetensors or $ZUNA_WEIGHTS");
            eprintln!("    * ZUNA config  at data/config.json         or $ZUNA_CONFIG");
            eprintln!("    * Python reference at data/python_reference.safetensors");
            eprintln!("      (produce with: python3 scripts/dump_python_reference.py …)");
            return;
        }
    };
    eprintln!("→ weights = {}", weights_path.display());
    eprintln!("→ config  = {}", config_path.display());
    eprintln!("→ pyref   = {}", ref_path.display());

    let ref_bytes = std::fs::read(&ref_path).expect("read reference");
    let st = SafeTensors::deserialize(&ref_bytes).expect("parse reference");

    let enc_input = read_f32_tensor(&st, "encoder_input");
    let tok_idx   = read_i32_tensor(&st, "tok_idx");
    let py_emb    = read_f32_tensor(&st, "embedding");
    let in_shape  = shape_of(&st, "encoder_input");
    let tok_shape = shape_of(&st, "tok_idx");
    let emb_shape = shape_of(&st, "embedding");

    assert_eq!(in_shape.len(), 2, "encoder_input must be [S, 32]");
    assert_eq!(tok_shape, vec![in_shape[0], 4], "tok_idx must be [S, 4]");
    assert_eq!(emb_shape.len(), 2, "embedding must be [S, output_dim]");
    let s = in_shape[0];
    let chan_pos = vec![0.0_f32; 3]; // unused by encoder graph; placeholder.

    let (mut rlx_enc, _) = ZunaEncoder::load(
        Path::new(config_path.to_str().unwrap()),
        Path::new(weights_path.to_str().unwrap()),
        pick_rlx_device(),
    ).expect("rlx encoder load");

    // tc and n_channels are bookkeeping for downstream decoding; for an
    // encoder-only parity we just pass plausible values.
    let rlx_emb_struct = rlx_enc.encode_one(
        &enc_input, &tok_idx, &chan_pos,
        /*n_channels=*/1, /*tc=*/s,
    ).expect("rlx encode_one");
    let rlx_emb = rlx_emb_struct.embeddings;

    assert_eq!(rlx_emb.len(), py_emb.len(),
        "output length mismatch: rlx={} python={}", rlx_emb.len(), py_emb.len());

    let mut max_abs = 0.0_f32;
    let mut sum_sq  = 0.0_f64;
    for (a, b) in rlx_emb.iter().zip(py_emb.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / rlx_emb.len() as f64).sqrt();
    let cos = cosine_similarity(&rlx_emb, &py_emb);

    eprintln!("→ rlx[0..6]: {:?}", &rlx_emb[..6.min(rlx_emb.len())]);
    eprintln!("→ py [0..6]: {:?}", &py_emb[..6.min(py_emb.len())]);
    eprintln!("→ parity:    max_abs={:.3e}  rms={:.3e}  cosine={:.6}  n={}",
        max_abs, rms, cos, rlx_emb.len());

    // The Python reference is f32 throughout, so the only noise source
    // here is reduction order (BLAS vs RLX SDPA kernel). Threshold is
    // tighter than the Burn-vs-RLX test for that reason.
    assert!(max_abs < 1e-3,
        "python parity failed: max_abs={max_abs:.3e} > 1e-3");
    assert!(cos > 0.9999,
        "cosine similarity too low: {cos:.6}");
}
