//! # ZUNA Embedding API — minimal example
//!
//! Demonstrates the three ways to produce EEG embeddings with `zuna-rs`.
//! No charts, no step-timers — just the core API calls.
//!
//! | Path | What it shows |
//! |---|---|
//! | A. One-shot   | [`ZunaEncoder::encode_fif`] — single call, get result, save |
//! | B. Two-step   | [`ZunaEncoder::preprocess_fif`] + [`ZunaEncoder::encode_batches`] |
//! | C. Per-tensor | [`ZunaEncoder::encode_tensor`] — raw Burn tensor output |
//!
//! ## Usage
//!
//! ```sh
//! # CPU (default), weights fetched/cached via hf-hub:
//! cargo run --example embedding_api --release --features hf-download
//!
//! # GPU (wgpu):
//! cargo run --example embedding_api --release \
//!     --no-default-features --features wgpu,hf-download \
//!     -- --device gpu
//!
//! # Point at local weight files instead (no network):
//! cargo run --example embedding_api --release -- \
//!     --weights model.safetensors --config config.json
//!
//! # Custom FIF recording:
//! cargo run --example embedding_api --release --features hf-download -- \
//!     --fif /path/to/my.fif
//! ```

use std::path::PathBuf;
use std::time::Instant;

use burn::prelude::{Backend, ElementConversion};
use clap::{Parser, ValueEnum};
use zuna_rs::{EpochEmbedding, EncodingResult, ZunaEncoder};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum Device { Cpu, Gpu }

#[derive(Parser, Debug)]
#[command(name = "embedding_api", about = "ZUNA EEG — minimal embedding API example")]
struct Args {
    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: Device,

    /// HuggingFace repo to download weights from (requires --features hf-download).
    #[arg(long, default_value = "Zyphra/ZUNA")]
    repo: String,

    /// Explicit safetensors weights file (skips HF download).
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Explicit config.json (must be paired with --weights).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Input EEG recording (.fif).
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif"))]
    fif: PathBuf,

    /// Output safetensors file for the embeddings produced by path A.
    #[arg(long, default_value = "embeddings.safetensors")]
    output: PathBuf,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point — dispatch to the compiled-in backend
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.device {
        Device::Cpu => run_cpu(args),
        Device::Gpu => run_gpu(args),
    }
}

#[cfg(feature = "ndarray")]
fn run_cpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    run::<NdArray>(NdArrayDevice::Cpu, args)
}
#[cfg(not(feature = "ndarray"))]
fn run_cpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("CPU backend not compiled — rebuild with `--features ndarray`")
}

#[cfg(feature = "wgpu")]
fn run_gpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{wgpu::WgpuDevice, Wgpu};
    run::<Wgpu>(WgpuDevice::DefaultDevice, args)
}
#[cfg(not(feature = "wgpu"))]
fn run_gpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("GPU backend not compiled — rebuild with `--no-default-features --features wgpu`")
}

// ─────────────────────────────────────────────────────────────────────────────
// Core — generic over any Burn backend
// ─────────────────────────────────────────────────────────────────────────────

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()> {
    // ── Resolve weights ───────────────────────────────────────────────────────
    let (weights, config) = resolve_weights(&args.repo, args.weights, args.config)?;
    println!("Weights : {}", weights.display());
    println!("Config  : {}", config.display());

    // ── Load encoder ──────────────────────────────────────────────────────────
    //
    // Only the encoder half of the weights is kept in RAM; decoder tensors
    // are read from disk once for key extraction then dropped.
    let t = Instant::now();
    let (encoder, _) = ZunaEncoder::<B>::load(&config, &weights, device)?;
    println!("Loaded  : {}  ({:.0} ms)\n", encoder.describe(), t.elapsed().as_secs_f64() * 1000.0);

    // Normalisation divisor — must match the value used at training time.
    let data_norm: f32 = 10.0;

    // =========================================================================
    // PATH A — one-shot  (simplest)
    // =========================================================================
    //
    // A single call that reads the .fif, splits it into 5-second epochs,
    // tokenises each epoch into (channel × time) tokens, and runs every epoch
    // through the encoder transformer.
    //
    // Returns an EncodingResult with one EpochEmbedding per epoch.
    println!("── A: one-shot encode ──────────────────────────────────────────");
    let result: EncodingResult = {
        let t = Instant::now();
        let r = encoder.encode_fif(&args.fif, data_norm)?;
        println!("encode_fif  : {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        print_result(&r);

        // Keys per epoch N: embeddings_N [n_tokens, dim], tok_idx_N [n_tokens, 4],
        //                   chan_pos_N   [n_channels, 3]
        // Plus: n_samples (scalar).
        r.save_safetensors(args.output.to_str().unwrap_or("embeddings.safetensors"))?;
        println!("Saved       : {}\n", args.output.display());
        r
    };
    let _ = result; // available for further use if needed

    // =========================================================================
    // PATH B — two-step  (preprocess then encode)
    // =========================================================================
    //
    // Split the pipeline so you can:
    //   • measure preprocessing and the forward pass separately
    //   • inspect / filter InputBatches before encoding
    //   • export pre-tokenised tensors for external comparison
    println!("── B: two-step (preprocess → encode) ───────────────────────────");
    {
        // Step 1: tokenise the FIF into one InputBatch per epoch.
        let t = Instant::now();
        let (batches, fif_info) = encoder.preprocess_fif(&args.fif, data_norm)?;
        println!(
            "preprocess  : {:.1} ms  │  {} epochs  │  {} ch  │  {:.0}→{:.0} Hz",
            t.elapsed().as_secs_f64() * 1000.0,
            batches.len(),
            fif_info.ch_names.len(),
            fif_info.sfreq,
            fif_info.target_sfreq,
        );

        // Step 2: run the transformer on all batches.
        let t = Instant::now();
        let epochs: Vec<EpochEmbedding> = encoder.encode_batches(batches)?;
        println!("encode      : {:.1} ms  │  {} epochs", t.elapsed().as_secs_f64() * 1000.0, epochs.len());

        if let Some(ep) = epochs.first() { print_epoch(0, ep); }
        println!();
    }

    // =========================================================================
    // PATH C — per-tensor  (raw Burn tensor)
    // =========================================================================
    //
    // Use this when the embedding feeds directly into further Burn operations:
    // downstream models, custom losses, normalisation, etc.
    // encode_tensor skips Vec allocation and returns a live Tensor<B, 3>.
    println!("── C: per-tensor (raw Burn tensor) ─────────────────────────────");
    {
        let (batches, _) = encoder.preprocess_fif(&args.fif, data_norm)?;

        if let Some(batch) = batches.into_iter().next() {
            // Tensor<B, 3>  shape = [1, n_tokens, output_dim]
            let t = Instant::now();
            let tensor = encoder.encode_tensor(&batch);
            println!("encode_tensor : {:.1} ms  │  shape {:?}", t.elapsed().as_secs_f64() * 1000.0, tensor.dims());

            // Compute mean and std on the live Burn tensor as an example.
            let flat   = tensor.flatten::<1>(0, 2);       // [n_tokens * output_dim]
            let mean_t = flat.clone().mean();              // scalar tensor
            let mean   = mean_t.clone().into_scalar().elem::<f32>();
            let diff   = flat - mean_t;
            let std    = (diff.clone() * diff).mean().into_scalar().elem::<f32>().sqrt();

            println!("  mean = {mean:+.4}  std = {std:.4}  (ideal ≈ 0.0 and ≈ 1.0 via MMD)");
        }
    }

    println!("\nDone.");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight resolution
// ─────────────────────────────────────────────────────────────────────────────

/// Return paths to (weights.safetensors, config.json).
///
/// Priority:
///   1. `--weights` + `--config` CLI args — explicit, offline-safe.
///   2. hf-hub (requires `--features hf-download`) — downloads on first run,
///      then returns the cached path instantly on subsequent runs.
fn resolve_weights(
    repo:    &str,
    weights: Option<PathBuf>,
    config:  Option<PathBuf>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    // Explicit paths — no network required.
    match (weights, config) {
        (Some(w), Some(c)) => return Ok((w, c)),
        (Some(_), None) | (None, Some(_)) =>
            anyhow::bail!("supply both --weights and --config together, or neither"),
        (None, None) => {}
    }

    // hf-hub: checks local cache first, downloads only if the file is missing.
    hf_download(repo)
}

/// Download (or return cached) weights via hf-hub.
///
/// Requires `--features hf-download`; compile stub bails with a clear message
/// when the feature is absent.
#[cfg(feature = "hf-download")]
fn hf_download(repo: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    use hf_hub::api::sync::ApiBuilder;
    let model   = ApiBuilder::new().with_progress(true).build()?.model(repo.to_string());
    let weights = model.get("model-00001-of-00001.safetensors")?;
    let config  = model.get("config.json")?;
    Ok((weights, config))
}

#[cfg(not(feature = "hf-download"))]
fn hf_download(_repo: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    anyhow::bail!(
        "Add `--features hf-download` to fetch weights automatically, \
         or pass --weights and --config explicitly."
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Print helpers
// ─────────────────────────────────────────────────────────────────────────────

fn print_result(r: &EncodingResult) {
    println!("  preproc : {:.1} ms  │  {} epochs", r.ms_preproc, r.epochs.len());
    println!("  encode  : {:.1} ms", r.ms_encode);
    if let Some(ep) = r.epochs.first() { print_epoch(0, ep); }
}

fn print_epoch(idx: usize, ep: &EpochEmbedding) {
    let n     = ep.embeddings.len();
    let mean: f64 = ep.embeddings.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let std:  f64 = (ep.embeddings.iter()
        .map(|&v| { let d = v as f64 - mean; d * d })
        .sum::<f64>() / n as f64).sqrt();
    let min = ep.embeddings.iter().copied().fold(f32::INFINITY,     f32::min);
    let max = ep.embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  epoch[{idx}]: {} tokens × {} dims  mean={mean:+.4}  std={std:.4}  [{min:+.3}, {max:+.3}]",
        ep.n_tokens(), ep.output_dim(),
    );
}
