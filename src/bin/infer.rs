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
/// Build — MLX (Apple Silicon native, macOS only):
///   cargo build --release --no-default-features --features mlx
///
/// Build — multiple backends for runtime selection:
///   cargo build --release --features ndarray,mlx
///
/// Usage:
///   infer --weights <st> --config <json> --fif <fif> --output <st>
///         [--device cpu|gpu|gpu-f16|mlx|mlx-f16]
///         [--steps 50] [--cfg 1.0] [--data-norm 10.0] [--verbose]

use std::{path::Path, time::Instant};
use burn::prelude::Backend;
use clap::{Parser, ValueEnum};
use zuna_rs::ZunaInference;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum Device { Cpu, Gpu, GpuF16, Mlx, MlxF16 }

#[derive(Parser, Debug)]
#[command(about = "ZUNA EEG model inference (Burn 0.20.1)")]
struct Args {
    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: Device,

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

    /// Number of CPU threads for NdArray backend (0 or omit = all cores).
    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    /// Print model config, electrode positions, per-epoch stats.
    #[arg(long, short = 'v')]
    verbose: bool,
}

// ── Per-backend shims ─────────────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
fn run_cpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    let name = if cfg!(feature = "blas-accelerate") { "CPU (NdArray + Apple Accelerate)" }
               else if cfg!(feature = "openblas-system") { "CPU (NdArray + OpenBLAS)" }
               else { "CPU (NdArray + Rayon)" };
    run::<NdArray>(NdArrayDevice::Cpu, name, args)
}
#[cfg(not(feature = "ndarray"))]
fn run_cpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("CPU backend not compiled — rebuild with `--features ndarray`")
}

#[cfg(any(feature = "wgpu", feature = "wgpu-f16"))]
fn run_gpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{wgpu::WgpuDevice, Wgpu};
    run::<Wgpu>(WgpuDevice::DefaultDevice, "GPU (wgpu f32)", args)
}
#[cfg(not(any(feature = "wgpu", feature = "wgpu-f16")))]
fn run_gpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("GPU backend not compiled — rebuild with `--features wgpu`")
}

#[cfg(any(feature = "wgpu-f16", feature = "wgpu"))]
fn run_gpu_f16(args: Args) -> anyhow::Result<()> {
    type B = burn::backend::wgpu::Wgpu<half::f16, i32, u32>;
    run::<B>(burn::backend::wgpu::WgpuDevice::DefaultDevice, "GPU (wgpu f16)", args)
}
#[cfg(not(any(feature = "wgpu-f16", feature = "wgpu")))]
fn run_gpu_f16(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("GPU f16 backend not compiled — rebuild with `--features wgpu-f16`")
}

#[cfg(any(feature = "mlx", feature = "mlx-f16"))]
fn run_mlx(args: Args) -> anyhow::Result<()> {
    use burn_mlx::{Mlx, MlxDevice};
    run::<Mlx>(MlxDevice::Gpu, "MLX (Apple Silicon f32)", args)
}
#[cfg(not(any(feature = "mlx", feature = "mlx-f16")))]
fn run_mlx(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("MLX backend not compiled — rebuild with `--features mlx`")
}

#[cfg(any(feature = "mlx-f16", feature = "mlx"))]
fn run_mlx_f16(args: Args) -> anyhow::Result<()> {
    use burn_mlx::{MlxHalf, MlxDevice};
    run::<MlxHalf>(MlxDevice::Gpu, "MLX (Apple Silicon f16)", args)
}
#[cfg(not(any(feature = "mlx-f16", feature = "mlx")))]
fn run_mlx_f16(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("MLX f16 backend not compiled — rebuild with `--features mlx-f16`")
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() -> anyhow::Result<()> {
    let args  = Args::parse();
    let _n_threads = zuna_rs::init_threads(args.threads);
    match args.device {
        Device::Cpu    => run_cpu(args),
        Device::Gpu    => run_gpu(args),
        Device::GpuF16 => run_gpu_f16(args),
        Device::Mlx    => run_mlx(args),
        Device::MlxF16 => run_mlx_f16(args),
    }
}

// ── Generic inference (backend-agnostic) ─────────────────────────────────────

fn run<B: Backend>(dev: B::Device, backend_name: &str, args: Args) -> anyhow::Result<()> {
    let n_threads = rayon::current_num_threads();
    let t0 = Instant::now();

    println!("Backend : {backend_name}  ({n_threads} threads)");

    // ── Load model ────────────────────────────────────────────────────────────
    let (zuna, ms_weights) = ZunaInference::<B>::load(
        Path::new(&args.config),
        Path::new(&args.weights),
        dev.clone(),
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

// Unused `device()` helper removed — device is now constructed per-backend in run_* shims.
