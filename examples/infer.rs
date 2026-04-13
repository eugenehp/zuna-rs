/// ZUNA EEG — full inference example.
///
/// Runs encode → diffuse → decode and saves reconstructed EEG to safetensors.
/// Weights are resolved automatically from HuggingFace cache by default.
///
/// # Usage
///
/// ```sh
/// # Default: CPU, bundled sample FIF, weights auto-resolved from HF cache
/// cargo run --example infer --release -- \
///     --weights model.safetensors --config config.json
///
/// # Or let it find the weights automatically (if already cached):
/// cargo run --example infer --release
///
/// # GPU (rebuild required), custom FIF, 10 fast steps:
/// cargo run --example infer --release \
///     --no-default-features --features wgpu -- \
///     --fif my.fif --device gpu --steps 10 --verbose
/// ```

#[path = "common/mod.rs"]
mod common;

use std::{path::Path, time::Instant};

use burn::prelude::Backend;
use clap::{Parser, ValueEnum};
use zuna_rs::ZunaInference;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum Device {
    /// CPU (NdArray + Rayon/BLAS).  Default, requires `--features ndarray`.
    Cpu,
    /// GPU (wgpu).  Requires `--no-default-features --features wgpu`.
    Gpu,
}

#[derive(Parser, Debug)]
#[command(
    name = "infer",
    about = "ZUNA EEG — full inference (encode + diffuse + decode)",
    after_help = "\
WEIGHT RESOLUTION (in order of priority)
  1. Both --weights and --config given explicitly.
  2. hf-hub download/cache  (requires feature `hf-download`).
  3. Scan ~/.cache/huggingface/hub/ for an existing snapshot.
  Use --repo to select a different HuggingFace model (default: Zyphra/ZUNA).

OUTPUT FILE  (safetensors)
  reconstructed_N  [n_channels, n_timesteps]  float32
  chan_pos_N       [n_channels, 3]             float32  (metres)
  n_samples        scalar                      float32

FIGURES  (written to --figures dir)
  infer_timing.png       Step-by-step wall-clock breakdown.
  infer_waveforms.png    First epoch reconstructed signal (up to 8 channels).
  infer_epoch_stats.png  Per-epoch mean / ±std / min-max summary.
  Pass --no-charts to skip chart generation.
"
)]
struct Args {
    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: Device,

    /// HuggingFace repo ID (used for automatic weight resolution).
    #[arg(long, default_value = common::DEFAULT_REPO, env = "ZUNA_REPO")]
    repo: String,

    /// HuggingFace cache directory override (default: ~/.cache/huggingface/hub).
    #[arg(long, env = "HF_HOME")]
    hf_cache: Option<std::path::PathBuf>,

    /// Explicit safetensors weights file (skip HF resolution).
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<String>,

    /// Explicit config.json (skip HF resolution; must pair with --weights).
    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<String>,

    /// Input EEG recording (.fif).
    #[arg(
        long,
        env = "ZUNA_FIF",
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif")
    )]
    fif: String,

    /// Output safetensors file.
    #[arg(long, default_value = "output.safetensors")]
    output: String,

    /// Directory to write performance charts.
    #[arg(long, default_value = "figures")]
    figures: String,

    /// Diffusion denoising steps (50 = full quality, 10 = fast preview).
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Classifier-free guidance scale (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    cfg: f32,

    /// Signal normalisation divisor (applied before model, inverted after).
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Verbose step-by-step output with per-epoch timing.
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Skip chart generation (useful in headless CI).
    #[arg(long)]
    no_charts: bool,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.device {
        Device::Cpu => run_cpu(args),
        Device::Gpu => run_gpu(args),
    }
}

// ── Per-backend shims ─────────────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
fn run_cpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    run::<NdArray>(NdArrayDevice::Cpu, cpu_name(), args)
}
#[cfg(not(feature = "ndarray"))]
fn run_cpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("CPU backend not compiled — rebuild with `--features ndarray`")
}

#[cfg(feature = "wgpu")]
fn run_gpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{wgpu::WgpuDevice, Wgpu};
    run::<Wgpu>(WgpuDevice::DefaultDevice, "GPU (wgpu / Metal or Vulkan)", args)
}
#[cfg(not(feature = "wgpu"))]
fn run_gpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("GPU backend not compiled — rebuild with `--no-default-features --features wgpu`")
}

fn cpu_name() -> &'static str {
    if cfg!(feature = "blas-accelerate") { "CPU (NdArray + Apple Accelerate)" }
    else if cfg!(feature = "openblas-system") { "CPU (NdArray + OpenBLAS)" }
    else { "CPU (NdArray + Rayon)" }
}

// ── Generic inference ─────────────────────────────────────────────────────────

fn run<B: Backend>(device: B::Device, backend_name: &str, args: Args) -> anyhow::Result<()> {
    let figures = std::path::PathBuf::from(&args.figures);
    if !args.no_charts { common::ensure_figures_dir(&figures)?; }

    // Total-time stopwatch (independent of StepTimer steps)
    let t_total = Instant::now();

    println!("Backend  : {backend_name}");
    println!("FIF      : {}", args.fif);

    // Steps: resolve → load → (preproc split into 2) → decode → save [→ charts]
    let total_steps = if args.no_charts { 5 } else { 6 };
    let mut timer = common::StepTimer::new(total_steps, args.verbose);
    let mut timing: Vec<(&'static str, f64)> = Vec::new();

    // ── Step 1: Resolve weights ───────────────────────────────────────────────
    timer.begin("Resolve weights");
    let (weights_path, config_path) = common::resolve_weights(
        &args.repo,
        args.weights.as_deref(),
        args.config.as_deref(),
        args.hf_cache.as_deref(),
    )?;
    let ms_resolve = timer.done(&format!(
        "weights={} config={}",
        weights_path.file_name().unwrap_or_default().to_string_lossy(),
        config_path.file_name().unwrap_or_default().to_string_lossy(),
    ));
    timing.push(("Resolve weights", ms_resolve));

    // ── Step 2: Load model ────────────────────────────────────────────────────
    timer.begin("Load model (encoder + decoder)");
    let (model, ms_weights) = ZunaInference::<B>::load(&config_path, &weights_path, device)?;
    let ms_load = timer.done(&model.describe());
    if args.verbose {
        timer.sub(&format!("  input_dim={} rope_theta={} σ={}",
            model.model_cfg.input_dim,
            model.model_cfg.rope_theta,
            model.model_cfg.stft_global_sigma));
    }
    timing.push(("Load model", ms_load));

    // ── Step 3: Preprocess + Encode ───────────────────────────────────────────
    timer.begin("Preprocess + encode (encoder forward)");
    let enc_result = model.encode_fif(Path::new(&args.fif), args.data_norm)?;
    let n_epochs = enc_result.epochs.len();
    let ms_enc = timer.done(&format!(
        "preproc={:.1}ms  encode={:.1}ms  epochs={}  tokens={}×{}",
        enc_result.ms_preproc, enc_result.ms_encode,
        n_epochs,
        enc_result.epochs.first().map(|e| e.n_tokens()).unwrap_or(0),
        enc_result.epochs.first().map(|e| e.output_dim()).unwrap_or(0),
    ));
    if args.verbose {
        if let Some(info) = &enc_result.fif_info {
            timer.sub(&format!("  channels={} sfreq={:.0}→{:.0} Hz  duration={:.2}s",
                info.ch_names.len(), info.sfreq, info.target_sfreq, info.duration_s));
            timer.sub(&format!("  preproc pipeline: resample→HP-FIR→avg-ref→zscore→epoch÷data_norm"));
            // Electrode table in verbose mode
            timer.sub("  Electrode positions (MNI, mm):");
            timer.sub(&format!("    {:<4} {:<10} {:>9} {:>9} {:>9}", "#", "Channel", "Right(x)", "Ant(y)", "Sup(z)"));
            for (i, (name, pos)) in info.ch_names.iter().zip(info.ch_pos_mm.iter()).enumerate() {
                timer.sub(&format!("    {:<4} {:<10} {:>9.2} {:>9.2} {:>9.2}",
                    i, name, pos[0], pos[1], pos[2]));
            }
        }
    }
    timing.push(("Preprocess", enc_result.ms_preproc));
    timing.push(("Encode", enc_result.ms_encode));
    let _ = ms_enc; // enc total is split into preproc + encode above

    // ── Step 4: Decode (diffusion) — per-epoch ────────────────────────────────
    timer.begin(&format!("Decode  ({n_epochs} epochs × {} diffusion steps)", args.steps));
    let mut dec_epochs = Vec::with_capacity(n_epochs);
    let mut ms_decode_total = 0.0f64;
    for (i, ep) in enc_result.epochs.iter().enumerate() {
        let t_ep = Instant::now();
        timer.sub(&format!("  epoch {}/{n_epochs}: {} tokens × {} dims → decode …",
            i + 1, ep.n_tokens(), ep.output_dim()));
        let out = model.decode_epoch(ep, args.steps, args.cfg, args.data_norm)?;
        let ms_ep = t_ep.elapsed().as_secs_f64() * 1000.0;
        ms_decode_total += ms_ep;
        if args.verbose {
            let d = &out.reconstructed;
            let mean: f64 = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
            let std: f64  = (d.iter().map(|&v| { let x = v as f64-mean; x*x })
                              .sum::<f64>() / d.len() as f64).sqrt();
            let min = d.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            timer.sub(&format!("    → {ms_ep:.0} ms  shape={:?}  mean={mean:+.3}  \
                                std={std:.3}  [{min:+.2}, {max:+.2}]", out.shape));
        }
        dec_epochs.push(out);
    }
    let ms_decode = timer.done(&format!("{n_epochs} epochs decoded  total={ms_decode_total:.0} ms"));
    timing.push(("Decode", ms_decode));

    // ── Step 5: Save output ───────────────────────────────────────────────────
    timer.begin("Save output");
    use zuna_rs::InferenceResult;
    let result = InferenceResult {
        epochs:     dec_epochs,
        fif_info:   enc_result.fif_info,
        ms_preproc: enc_result.ms_preproc,
        ms_infer:   ms_decode_total,
    };
    result.save_safetensors(&args.output)?;
    let ms_save = timer.done(&format!("→ {}", args.output));
    timing.push(("Save", ms_save));

    // ── Step 6: Charts ────────────────────────────────────────────────────────
    if !args.no_charts {
        timer.begin("Generate charts");

        // 6a. Timing breakdown
        common::save_timing_chart(
            &figures.join("infer_timing.png"),
            &format!("ZUNA Inference — timing  ({backend_name})"),
            &timing.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        ).unwrap_or_else(|e| eprintln!("⚠  timing chart: {e}"));

        // 6b. First epoch waveform
        if let Some(ep) = result.epochs.first() {
            let n_ch  = ep.n_channels;
            let n_t   = ep.shape.get(1).copied().unwrap_or(0);
            let sfreq = result.fif_info.as_ref().map(|f| f.target_sfreq).unwrap_or(256.0);
            let ch_names: Vec<String> = result.fif_info.as_ref()
                .map(|f| f.ch_names.clone())
                .unwrap_or_else(|| (0..n_ch).map(|i| format!("Ch{i}")).collect());

            // Reshape flat [n_ch * n_t] → Vec of Vec per channel
            let channels: Vec<Vec<f32>> = (0..n_ch)
                .map(|c| ep.reconstructed[c * n_t..(c + 1) * n_t].to_vec())
                .collect();

            common::save_waveform_chart(
                &figures.join("infer_waveforms.png"),
                "ZUNA — Epoch 1 reconstructed EEG signal",
                &channels,
                &ch_names,
                sfreq,
            ).unwrap_or_else(|e| eprintln!("⚠  waveform chart: {e}"));
        }

        // 6c. Per-epoch stats
        let (means, stds, mins, maxs): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) = result.epochs
            .iter()
            .map(|ep| {
                let d = &ep.reconstructed;
                let mean: f64 = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
                let std: f64  = (d.iter().map(|&v| { let x = v as f64 - mean; x*x })
                                  .sum::<f64>() / d.len() as f64).sqrt();
                let min = d.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
                let max = d.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                (mean, std, min, max)
            })
            .unzip4();

        common::save_epoch_stats_chart(
            &figures.join("infer_epoch_stats.png"),
            "ZUNA — Per-epoch reconstruction statistics",
            &means, &stds, &mins, &maxs,
        ).unwrap_or_else(|e| eprintln!("⚠  epoch-stats chart: {e}"));

        timer.done(&format!("charts → {}/", args.figures));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let ms_total = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── Summary ────────────────────────────────────────────────");
    println!("  Weights  : {ms_weights:.0} ms");
    println!("  Preproc  : {:.1} ms", enc_result.ms_preproc);
    println!("  Encode   : {:.1} ms  ({n_epochs} epochs)", enc_result.ms_encode);
    println!("  Decode   : {ms_decode_total:.0} ms  ({n_epochs} epochs × {} steps)", args.steps);
    println!("  Total    : {ms_total:.0} ms");
    println!("  Output   : {}", args.output);
    eprintln!("TIMING weights={ms_weights:.1}ms preproc={:.1}ms \
               encode={:.1}ms decode={ms_decode_total:.1}ms total={ms_total:.1}ms",
              enc_result.ms_preproc, enc_result.ms_encode);

    Ok(())
}

// ── Helper: unzip 4-tuple iterator ───────────────────────────────────────────

trait Unzip4<A, B, C, D> {
    fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>);
}
impl<I, A, B, C, D> Unzip4<A, B, C, D> for I
where
    I: Iterator<Item = (A, B, C, D)>,
{
    fn unzip4(self) -> (Vec<A>, Vec<B>, Vec<C>, Vec<D>) {
        let mut va = Vec::new(); let mut vb = Vec::new();
        let mut vc = Vec::new(); let mut vd = Vec::new();
        for (a, b, c, d) in self { va.push(a); vb.push(b); vc.push(c); vd.push(d); }
        (va, vb, vc, vd)
    }
}
