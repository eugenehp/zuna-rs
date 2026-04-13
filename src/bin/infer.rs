/// ZUNA EEG inference — thin CLI over [`zuna_rs::ZunaInference`].
///
/// All model logic lives in `src/inference.rs`.  This file is just argument
/// parsing, display, and I/O.
///
/// Build — CPU (default; Apple Accelerate on macOS):
///   cargo build --release [--features blas-accelerate]
///
/// Build — GPU (Metal on macOS, Vulkan on Linux):
///   cargo build --release --no-default-features --features wgpu
///
/// Usage:
///   infer --weights <st> --config <json> --fif <fif> --output <st>
///         [--steps 50] [--cfg 1.0] [--data-norm 10.0] [--verbose]

use std::{path::Path, time::Instant};
use clap::Parser;
use zuna_rs::ZunaInference;

// ── Backend ───────────────────────────────────────────────────────────────────
#[cfg(all(feature = "wgpu", not(feature = "ndarray")))]
mod backend {
    pub use burn::backend::{Wgpu as B, wgpu::WgpuDevice as Device};
    pub fn device() -> Device { Device::DefaultDevice }
    pub const NAME: &str = "GPU (wgpu / Metal or Vulkan)";
}

#[cfg(feature = "ndarray")]
mod backend {
    pub use burn::backend::NdArray as B;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub fn device() -> Device { Device::Cpu }
    #[cfg(feature = "blas-accelerate")]
    pub const NAME: &str = "CPU (NdArray + Apple Accelerate)";
    #[cfg(feature = "openblas-system")]
    pub const NAME: &str = "CPU (NdArray + OpenBLAS)";
    #[cfg(not(any(feature = "blas-accelerate", feature = "openblas-system")))]
    pub const NAME: &str = "CPU (NdArray + Rayon)";
}

use backend::{B, device};

// ── CLI ───────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(about = "ZUNA EEG model inference (Burn 0.20.1)")]
struct Args {
    /// Safetensors weights file (from HuggingFace Zyphra/ZUNA).
    #[arg(long)]
    weights: String,

    /// config.json from HuggingFace Zyphra/ZUNA.
    #[arg(long)]
    config: String,

    /// Raw EEG recording (.fif).  Exactly one of --fif / --input required.
    #[arg(long)]
    fif: Option<String>,

    /// Pre-processed safetensors batch (legacy Python path).
    #[arg(long)]
    input: Option<String>,

    /// Output safetensors file.
    #[arg(long)]
    output: String,

    /// Diffusion denoising steps (50 = full quality, 10 = fast preview).
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Classifier-free guidance scale (1.0 = off).
    #[arg(long, default_value_t = 1.0)]
    cfg: f32,

    /// Signal normalisation divisor (applied before model, inverted after).
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Print model config, electrode positions, per-epoch stats.
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() -> anyhow::Result<()> {
    let args  = Args::parse();
    let t0    = Instant::now();
    let dev   = device();

    println!("Backend : {}", backend::NAME);

    // ── Load model ────────────────────────────────────────────────────────────
    let (zuna, ms_weights) = ZunaInference::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        dev,
    )?;

    if args.verbose {
        println!("── Model ─────────────────────────────────────────────────────────");
        println!("  {}", zuna.describe());
        println!("  input_dim  : {}", zuna.model_cfg.input_dim);
        println!("  rope_theta : {}", zuna.model_cfg.rope_theta);
        println!("  Loaded in {ms_weights:.0} ms");
    } else {
        println!("Model   : {}  ({ms_weights:.0} ms)", zuna.describe());
    }

    // ── Run pipeline ──────────────────────────────────────────────────────────
    let result = match (&args.fif, &args.input) {
        (Some(fif_path), None) => {
            println!("Input   : {fif_path}");
            let r = zuna.run_fif(
                Path::new(fif_path),
                args.steps,
                args.cfg,
                args.data_norm,
            )?;

            if args.verbose {
                let info = r.fif_info.as_ref().unwrap();

                println!("── FIF ───────────────────────────────────────────────────────────");
                println!("  Channels  : {}", info.ch_names.len());
                println!("  Sfreq     : {:.1} Hz  →  {:.1} Hz", info.sfreq, info.target_sfreq);
                println!("  Duration  : {:.3} s  ({} raw samples)", info.duration_s, info.n_times_raw);
                println!("  Epochs    : {} × {:.1} s  ({} samples each)",
                    info.n_epochs, info.epoch_dur_s,
                    (info.epoch_dur_s * info.target_sfreq) as usize);
                println!("  Preproc   : {:.1} ms", r.ms_preproc);

                println!("── Electrode positions (MNI head frame, mm) ──────────────────────");
                println!("  {:<4} {:<8} {:>10} {:>10} {:>10}", "#", "Name", "Right(x)", "Ant(y)", "Sup(z)");
                println!("  {}", "─".repeat(46));
                for (i, (name, pos)) in info.ch_names.iter().zip(info.ch_pos_mm.iter()).enumerate() {
                    println!("  {:<4} {:<8} {:>10.2} {:>10.2} {:>10.2}",
                        i, name, pos[0], pos[1], pos[2]);
                }
            } else {
                let info = r.fif_info.as_ref().unwrap();
                println!("  Preproc   : {:.1} ms  ({} epochs)", r.ms_preproc, info.n_epochs);
            }
            r
        }
        (None, Some(input_path)) => {
            println!("Input   : {input_path}  (safetensors batch)");
            zuna.run_safetensors_batch(
                Path::new(input_path),
                args.steps,
                args.cfg,
                args.data_norm,
            )?
        }
        (Some(_), Some(_)) => anyhow::bail!("supply exactly one of --fif or --input"),
        (None, None)        => anyhow::bail!("--fif or --input is required"),
    };

    // ── Per-epoch output ──────────────────────────────────────────────────────
    let n = result.epochs.len();
    println!("Epochs  : {n}  ({} steps  cfg={:.2})", args.steps, args.cfg);

    for (i, ep) in result.epochs.iter().enumerate() {
        if args.verbose {
            let data = &ep.reconstructed;
            let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
            let std:  f64 = (data.iter().map(|&v| {
                let d = v as f64 - mean; d*d
            }).sum::<f64>() / data.len() as f64).sqrt();
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  [ep {}/{}] {:?}  mean={mean:.4}  std={std:.4}  \
                      min={min:.4}  max={max:.4}",
                i+1, n, ep.shape);
        } else {
            println!("  [ep {}/{n}] {:?}  {:.0} ms", i+1, ep.shape, result.ms_infer / n as f64);
        }
    }

    // ── Timing ───────────────────────────────────────────────────────────────
    let ms_total = t0.elapsed().as_secs_f64() * 1000.0;
    println!("── Timing ───────────────────────────────────────────────────────");
    println!("  Weights  : {ms_weights:.0} ms");
    println!("  Preproc  : {:.1} ms", result.ms_preproc);
    println!("  Infer    : {:.0} ms  ({n} × {} steps)", result.ms_infer, args.steps);
    println!("  Total    : {ms_total:.0} ms");
    // Machine-readable timing for shell capture
    eprintln!("TIMING weights={ms_weights:.1}ms preproc={:.1}ms inference={:.1}ms total={ms_total:.1}ms",
              result.ms_preproc, result.ms_infer);

    // ── Save ──────────────────────────────────────────────────────────────────
    result.save_safetensors(&args.output)?;
    println!("Output  → {}", args.output);

    Ok(())
}
