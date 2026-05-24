//! Parity test: RLX-backed `ZunaEncoder` vs the original Burn-backed
//! reference implementation.
//!
//! Runs the same single epoch through both backends with bit-identical
//! weights and asserts the encoder output matches within tolerance
//! (`max abs diff < 5e-3` by default; bf16 → f32 round-trip plus
//! different reduction order is the dominant noise source).
//!
//! ## Running
//!
//! The test needs a HuggingFace ZUNA snapshot on disk. Either:
//!
//! ```text
//! cargo run --release --bin download_weights --features hf-download
//! ```
//!
//! then set `ZUNA_WEIGHTS` and `ZUNA_CONFIG` to the printed paths, **or**
//! place them in the workspace as `data/model.safetensors` and
//! `data/config.json`.
//!
//! Without weights the test is skipped with a friendly message (printed
//! with `--nocapture`).
//!
//! ```text
//! cargo test --release --no-default-features \
//!     --features ndarray,rlx-backend,rlx-cpu \
//!     --test parity_rlx_vs_burn -- --nocapture
//! ```

#![cfg(feature = "rlx-backend")]

use std::path::{Path, PathBuf};

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use zuna_rs::data::preprocess_fif_cpu;
use zuna_rs::rlx::ZunaEncoder as RlxEncoder;
use zuna_rs::{config::DataConfig, ZunaEncoder as BurnEncoder};

type B = NdArray<f32>;

/// Pick the RLX device based on the `ZUNA_RLX_DEVICE` env var.
/// Defaults to CPU. Recognised: `cpu`, `metal`, `mlx`.
fn pick_rlx_device() -> rlx::Device {
    match std::env::var("ZUNA_RLX_DEVICE").as_deref().unwrap_or("cpu") {
        "cpu"   => rlx::Device::Cpu,
        "metal" => rlx::Device::Metal,
        "mlx"   => rlx::Device::Mlx,
        other   => panic!("unknown ZUNA_RLX_DEVICE: {other:?}"),
    }
}

fn locate_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let weights = std::env::var("ZUNA_WEIGHTS").ok().map(PathBuf::from)
        .or_else(|| {
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/model.safetensors");
            p.exists().then_some(p)
        })?;
    let config = std::env::var("ZUNA_CONFIG").ok().map(PathBuf::from)
        .or_else(|| {
            let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/config.json");
            p.exists().then_some(p)
        })?;
    let fif = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/sample1_raw.fif");
    if !fif.exists() { return None; }
    Some((weights, config, fif))
}

#[test]
fn rlx_encoder_matches_burn_encoder() {
    let (weights, config, fif) = match locate_paths() {
        Some(t) => t,
        None => {
            eprintln!("\n[SKIP] parity test — missing weights/config/sample.");
            eprintln!("       expected ZUNA_WEIGHTS / ZUNA_CONFIG env vars,");
            eprintln!("       or data/{{model.safetensors,config.json,sample1_raw.fif}}");
            return;
        }
    };
    eprintln!("→ weights = {}", weights.display());
    eprintln!("→ config  = {}", config.display());
    eprintln!("→ fif     = {}", fif.display());

    // 1. CPU preprocessing — backend-agnostic.
    let data_cfg = DataConfig::default();
    let pre = preprocess_fif_cpu(&fif, &data_cfg, 10.0)
        .expect("preprocess_fif_cpu");
    assert!(!pre.epochs.is_empty(), "expected at least one epoch");
    let ep = &pre.epochs[0];
    eprintln!("→ epoch 0: s={} tf={} n_channels={}", ep.s, ep.tf, ep.n_channels);

    // 2. Burn reference.
    let (burn_enc, _) = BurnEncoder::<B>::load(
        Path::new(config.to_str().unwrap()),
        Path::new(weights.to_str().unwrap()),
        NdArrayDevice::Cpu,
    ).expect("burn encoder load");
    let burn_batches = burn_enc.preprocess_fif(&fif, 10.0)
        .expect("burn preprocess_fif").0;
    let burn_embeddings = burn_enc.encode_batches(burn_batches)
        .expect("burn encode_batches");
    assert_eq!(burn_embeddings.len(), pre.epochs.len(),
        "burn epoch count mismatch");
    let burn_first = &burn_embeddings[0];
    eprintln!("→ burn[0]: shape={:?} first8={:?}",
        burn_first.shape, &burn_first.embeddings[..8.min(burn_first.embeddings.len())]);

    // 3. RLX implementation.
    let (mut rlx_enc, _) = RlxEncoder::load(
        Path::new(config.to_str().unwrap()),
        Path::new(weights.to_str().unwrap()),
        pick_rlx_device(),
    ).expect("rlx encoder load");
    let rlx_emb = rlx_enc.encode_one(
        &ep.eeg_tokens, &ep.tok_idx, &ep.chan_pos,
        ep.n_channels, ep.tc,
    ).expect("rlx encode_one");
    eprintln!("→ rlx[0]:  shape={:?} first8={:?}",
        rlx_emb.shape, &rlx_emb.embeddings[..8.min(rlx_emb.embeddings.len())]);

    // 4. Compare.
    assert_eq!(burn_first.shape, rlx_emb.shape,
        "encoder output shape mismatch");
    let n = burn_first.embeddings.len();
    assert_eq!(n, rlx_emb.embeddings.len());
    let mut max_abs = 0.0_f32;
    let mut sum_sq = 0.0_f64;
    for (a, b) in burn_first.embeddings.iter().zip(rlx_emb.embeddings.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        sum_sq += (d as f64) * (d as f64);
    }
    let rms = (sum_sq / n as f64).sqrt();
    eprintln!("→ parity:  max_abs={:.3e}  rms={:.3e}  n={}", max_abs, rms, n);

    // Tolerance is intentionally generous on the first pass — bf16
    // weights + different reduction order between the Burn and RLX
    // paths can easily produce 1e-3 deviations. Tighten as the parity
    // hardens.
    assert!(max_abs < 5e-3,
        "parity failed: max_abs={max_abs:.3e} > 5e-3");
}
