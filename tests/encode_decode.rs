//! Integration tests for the encoder/decoder split API.
//!
//! These tests load real weights and FIF data from the paths available in the
//! CI environment.  They are skipped automatically when the files are absent
//! (run `cargo test --release` after the weights have been downloaded).

use std::path::{Path, PathBuf};

// CPU backend only (no GPU in test environment)
use burn::backend::NdArray as B;
type Device = burn::backend::ndarray::NdArrayDevice;

fn device() -> Device { Device::Cpu }

/// Return (config, weights, fif) paths, or skip the test if any are missing.
fn test_paths() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let hf_cache = PathBuf::from("/root/.cache/huggingface/hub/models--Zyphra--ZUNA/snapshots/local");
    let weights  = hf_cache.join("model-00001-of-00001.safetensors");
    let config   = hf_cache.join("config.json");
    let fif      = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/sample1_raw.fif");

    if weights.exists() && config.exists() && fif.exists() {
        Some((config, weights, fif))
    } else {
        None
    }
}

// ── ZunaEncoder standalone ────────────────────────────────────────────────────

#[test]
fn encoder_only_load_and_encode_fif() {
    let Some((config, weights, fif)) = test_paths() else {
        eprintln!("SKIP: weights or FIF not present");
        return;
    };

    use zuna_rs::ZunaEncoder;

    let (enc, ms_load) = ZunaEncoder::<B>::load(&config, &weights, device())
        .expect("encoder load");
    println!("Encoder loaded in {ms_load:.0} ms");
    println!("{}", enc.describe());

    let result = enc.encode_fif(&fif, 10.0).expect("encode_fif");

    assert!(!result.epochs.is_empty(), "must produce at least one epoch");

    for (i, ep) in result.epochs.iter().enumerate() {
        // Shape: [n_tokens, output_dim]
        assert_eq!(ep.shape.len(), 2);
        let [n_tok, out_dim] = [ep.shape[0], ep.shape[1]];
        assert_eq!(n_tok, ep.n_channels * ep.tc);
        assert_eq!(out_dim, 32, "ZUNA encoder_output_dim = 32");

        // Correct number of elements
        assert_eq!(ep.embeddings.len(), n_tok * out_dim);
        assert_eq!(ep.tok_idx.len(),   n_tok * 4);
        assert_eq!(ep.chan_pos.len(),   ep.n_channels * 3);

        // MMD regularisation: embeddings should be approximately zero-centred.
        // With only ~480 tokens we just check they're not all zero / NaN.
        let any_nonzero = ep.embeddings.iter().any(|&v| v != 0.0);
        let any_nan     = ep.embeddings.iter().any(|v| v.is_nan());
        assert!(any_nonzero, "epoch {i}: embeddings are all zero");
        assert!(!any_nan,     "epoch {i}: embeddings contain NaN");

        // Rough sanity: mean absolute value should be < 5 (MMD → ≈N(0,1)).
        let mean_abs: f32 = ep.embeddings.iter().map(|v| v.abs()).sum::<f32>()
            / ep.embeddings.len() as f32;
        assert!(mean_abs < 5.0,
            "epoch {i}: mean |embedding| = {mean_abs:.3}, unexpectedly large");

        println!("  epoch {i}: tokens={n_tok} out_dim={out_dim} mean_abs={mean_abs:.3}");
    }

    // Saving to safetensors must not panic.
    let out_path = "/tmp/zuna_test_embeddings.safetensors";
    result.save_safetensors(out_path).expect("save embeddings");
    assert!(Path::new(out_path).exists());
    println!("  saved → {out_path}");
}

// ── ZunaDecoder standalone (decode from stored embeddings) ───────────────────

#[test]
fn decoder_only_load_and_decode_embeddings() {
    let Some((config, weights, fif)) = test_paths() else {
        eprintln!("SKIP: weights or FIF not present");
        return;
    };

    use zuna_rs::{ZunaEncoder, ZunaDecoder};

    // 1. Encode with encoder-only
    let (enc, _) = ZunaEncoder::<B>::load(&config, &weights, device())
        .expect("encoder load");
    let embeddings = enc.encode_fif(&fif, 10.0).expect("encode_fif");

    // 2. Decode with decoder-only
    let (dec, ms_load) = ZunaDecoder::<B>::load(&config, &weights, device())
        .expect("decoder load");
    println!("Decoder loaded in {ms_load:.0} ms");
    println!("{}", dec.describe());

    let result = dec.decode_embeddings(&embeddings, /*steps=*/2, /*cfg=*/1.0, /*data_norm=*/10.0)
        .expect("decode_embeddings");

    assert_eq!(result.epochs.len(), embeddings.epochs.len(),
               "epoch count must match");

    for (i, ep) in result.epochs.iter().enumerate() {
        let [n_ch, n_t] = [ep.shape[0], ep.shape[1]];
        assert_eq!(n_ch, embeddings.epochs[i].n_channels);
        assert_eq!(n_t, 1280, "5 s × 256 Hz = 1280 samples");
        assert_eq!(ep.reconstructed.len(), n_ch * n_t);

        let any_nan = ep.reconstructed.iter().any(|v| v.is_nan());
        assert!(!any_nan, "epoch {i}: reconstruction contains NaN");

        println!("  epoch {i}: shape={:?}", ep.shape);
    }
}

// ── ZunaInference::encode_fif convenience method ─────────────────────────────

#[test]
fn full_model_encode_fif() {
    let Some((config, weights, fif)) = test_paths() else {
        eprintln!("SKIP: weights or FIF not present");
        return;
    };

    use zuna_rs::ZunaInference;

    let (model, _) = ZunaInference::<B>::load(&config, &weights, device())
        .expect("full model load");

    // Encode without decoding
    let enc_result = model.encode_fif(&fif, 10.0).expect("encode_fif");
    assert!(!enc_result.epochs.is_empty());

    // Embeddings should match the standalone encoder's output shape
    for ep in &enc_result.epochs {
        assert_eq!(ep.shape[1], 32, "output_dim must be 32");
    }

    // Full pipeline still works after encode-only call
    let infer_result = model.run_fif(&fif, 2, 1.0, 10.0)
        .expect("run_fif");
    assert_eq!(infer_result.epochs.len(), enc_result.epochs.len());

    println!("encode_fif and run_fif produced {} epochs each",
             enc_result.epochs.len());
}
