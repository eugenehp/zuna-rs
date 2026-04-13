/// ZUNA EEG — encoder-only embeddings example.
///
/// Runs the encoder and saves per-epoch latent embeddings to safetensors.
/// Weights are resolved automatically from HuggingFace cache by default
/// (downloads via python3 huggingface_hub if not cached).
///
/// # Usage
///
/// ```sh
/// # Minimal — auto-resolves weights from HF cache (downloads if missing)
/// cargo run --example embed --release
///
/// # Explicit weights (useful in offline environments):
/// cargo run --example embed --release -- \
///     --weights model.safetensors --config config.json
///
/// # Export encoder inputs for Python comparison:
/// cargo run --example embed --release -- --export-inputs bench_inputs.safetensors
///
/// # GPU, verbose, custom FIF:
/// cargo run --example embed --release \
///     --no-default-features --features wgpu -- \
///     --device gpu --fif my.fif --verbose
/// ```

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use burn::prelude::Backend;
use clap::{Parser, ValueEnum};
use zuna_rs::{ZunaEncoder, data::InputBatch};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum Device { Cpu, Gpu }

#[derive(Parser, Debug)]
#[command(
    name = "embed",
    about = "ZUNA EEG — encoder-only embeddings with MMD statistics",
    after_help = "\
WEIGHT RESOLUTION (priority order)
  1. Both --weights and --config given explicitly.
  2. hf-hub download/cache  (feature `hf-download`).
  3. Scan ~/.cache/huggingface/hub/ for an existing snapshot.
  4. Download via `python3 -c \"from huggingface_hub import snapshot_download; ...\"`.
  Use --repo to select a different HuggingFace model (default: Zyphra/ZUNA).

OUTPUT FILES
  embeddings.safetensors (or --output):
    embeddings_N   [n_tokens, output_dim]  f32   — encoder latent
    tok_idx_N      [n_tokens, 4]           i64   — token metadata
    chan_pos_N     [n_channels, 3]         f32   — electrode positions
    n_samples      scalar                  f32

  bench_inputs.safetensors (--export-inputs):
    encoder_input_N  [n_tokens, 32]  f32   — pre-transformer tokenised EEG
    tok_idx_N        [n_tokens, 4]   i64   — token positions
    n_epochs         scalar          f32
    Used by bench_and_visualize.py to compare Python vs Rust encoder.

FIGURES  (--figures dir)
  embed_timing.png        Wall-clock breakdown.
  embed_distribution.png  Embedding histogram vs N(0,1).
  embed_dim_stats.png     Per-dimension mean ± std.
  Skip with --no-charts.
"
)]
struct Args {
    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: Device,

    /// HuggingFace repo ID for automatic weight resolution.
    #[arg(long, default_value = common::DEFAULT_REPO, env = "ZUNA_REPO")]
    repo: String,

    /// HuggingFace cache directory override.
    #[arg(long, env = "HF_HOME")]
    hf_cache: Option<std::path::PathBuf>,

    /// Explicit safetensors weights file (skip HF resolution).
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<String>,

    /// Explicit config.json (must pair with --weights).
    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<String>,

    /// Input EEG recording (.fif).
    #[arg(
        long, env = "ZUNA_FIF",
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif")
    )]
    fif: String,

    /// Output safetensors file for embeddings.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/embeddings.safetensors"))]
    output: String,

    /// Export raw encoder inputs (encoder_input + tok_idx) to this path.
    /// Used by bench_and_visualize.py for Python-vs-Rust comparison.
    #[arg(long)]
    export_inputs: Option<String>,

    /// Directory to write performance charts.
    #[arg(long, default_value = "figures")]
    figures: String,

    /// Signal normalisation divisor.
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Verbose step-by-step output with per-epoch timing and MMD statistics.
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Skip chart generation.
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
    run::<Wgpu>(WgpuDevice::DefaultDevice, "GPU (wgpu)", args)
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

// ── Generic encoder run ───────────────────────────────────────────────────────

fn run<B: Backend>(device: B::Device, backend_name: &str, args: Args) -> anyhow::Result<()> {
    let figures = std::path::PathBuf::from(&args.figures);
    if !args.no_charts { common::ensure_figures_dir(&figures)?; }

    let t_total = Instant::now();
    println!("Backend  : {backend_name}");
    println!("FIF      : {}", args.fif);

    // Steps: resolve → load → preprocess → encode → save [→ export-inputs] [→ charts]
    let export     = args.export_inputs.is_some();
    let total_steps = 5 + (!args.no_charts as usize) + (export as usize);
    let mut timer  = common::StepTimer::new(total_steps, args.verbose);
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
        "weights={}  config={}",
        weights_path.file_name().unwrap_or_default().to_string_lossy(),
        config_path.file_name().unwrap_or_default().to_string_lossy(),
    ));
    timing.push(("Resolve weights", ms_resolve));

    // ── Step 2: Load encoder ──────────────────────────────────────────────────
    timer.begin("Load encoder");
    let (encoder, ms_weights) = ZunaEncoder::<B>::load(&config_path, &weights_path, device)?;
    let ms_load = timer.done(&encoder.describe());
    timing.push(("Load encoder", ms_load));

    // ── Step 3: Preprocess FIF ────────────────────────────────────────────────
    timer.begin("Preprocess FIF");
    let t_pp = Instant::now();
    let (batches, fif_info) = encoder.preprocess_fif(Path::new(&args.fif), args.data_norm)?;
    let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;
    let n_epochs = batches.len();
    let (n_ch, n_tok) = batches.first()
        .map(|b| (b.n_channels, b.n_channels * b.tc))
        .unwrap_or((0, 0));
    timer.done(&format!(
        "{ms_preproc:.1} ms  {n_epochs} epochs  channels={n_ch}  tokens/ep={n_tok}  \
         sfreq={:.0}→{:.0} Hz  dur={:.2}s",
        fif_info.sfreq, fif_info.target_sfreq, fif_info.duration_s,
    ));
    timing.push(("Preprocess", ms_preproc));
    if args.verbose {
        timer.sub(&format!("  channels: {}", fif_info.ch_names.join(", ")));
    }

    // ── Step 4: Encode ────────────────────────────────────────────────────────
    timer.begin("Encode (encoder forward pass)");
    // Clone references before consuming batches
    let input_dim = batches.first().map(|b| {
        let [_, _, d] = b.encoder_input.dims();
        d
    }).unwrap_or(0);
    let t_enc = Instant::now();
    let epochs = encoder.encode_batches(batches)?;
    let ms_encode = t_enc.elapsed().as_secs_f64() * 1000.0;
    let out_dim = epochs.first().map(|e| e.output_dim()).unwrap_or(0);
    timer.done(&format!(
        "{ms_encode:.1} ms  {n_epochs} epochs  \
         tokens×dims = {n_tok}×{out_dim}  input_dim={input_dim}"
    ));
    timing.push(("Encode", ms_encode));

    if args.verbose {
        for (i, ep) in epochs.iter().enumerate() {
            let emb = &ep.embeddings;
            let mean: f64 = emb.iter().map(|&v| v as f64).sum::<f64>() / emb.len() as f64;
            let std: f64  = (emb.iter().map(|&v| { let x = v as f64 - mean; x*x })
                              .sum::<f64>() / emb.len() as f64).sqrt();
            let min = emb.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = emb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            timer.sub(&format!("  epoch {i}: tokens={} dims={}  \
                                mean={mean:+.4}  std={std:.4}  [{min:+.3},{max:+.3}]",
                ep.n_tokens(), ep.output_dim()));
        }
    }

    // Wrap into EncodingResult for saving
    use zuna_rs::encoder::EncodingResult;
    let result = EncodingResult { epochs, fif_info: Some(fif_info), ms_preproc, ms_encode };

    // ── Step 5: Save embeddings ───────────────────────────────────────────────
    timer.begin("Save embeddings");
    result.save_safetensors(&args.output)?;
    let ms_save = timer.done(&format!("→ {}", args.output));
    timing.push(("Save", ms_save));

    // ── Optional: Export encoder inputs for Python comparison ─────────────────
    if let Some(ref inputs_path) = args.export_inputs {
        timer.begin("Export encoder inputs (for bench_and_visualize.py)");
        // Re-preprocess to get the raw InputBatches (avoids lifetime complexity)
        let (batches2, _) = encoder.preprocess_fif(Path::new(&args.fif), args.data_norm)?;
        export_encoder_inputs::<B>(&batches2, inputs_path)?;
        timer.done(&format!("→ {inputs_path}  ({} epochs)", batches2.len()));
    }

    // ── Charts ────────────────────────────────────────────────────────────────
    if !args.no_charts {
        timer.begin("Generate charts");

        let all_vals: Vec<f32> = result.epochs.iter()
            .flat_map(|ep| ep.embeddings.iter().copied())
            .collect();

        let n_dims = result.epochs.first().map(|e| e.output_dim()).unwrap_or(0);
        let (dim_means, dim_stds) = if n_dims > 0 {
            compute_dim_stats(&result.epochs, n_dims)
        } else { (vec![], vec![]) };

        if args.verbose && !dim_means.is_empty() {
            let gmean = dim_means.iter().sum::<f64>() / dim_means.len() as f64;
            let gstd  = dim_stds.iter().sum::<f64>()  / dim_stds.len()  as f64;
            timer.sub(&format!("  MMD check — dim-avg mean={gmean:+.4}  dim-avg std={gstd:.4}  \
                               (ideal: 0.0 and 1.0)"));
        }

        common::save_timing_chart(
            &figures.join("embed_timing.png"),
            &format!("ZUNA Embeddings — timing  ({backend_name})"),
            &timing.iter().map(|(l, v)| (*l, *v)).collect::<Vec<_>>(),
        ).unwrap_or_else(|e| eprintln!("⚠  timing chart: {e}"));

        if !all_vals.is_empty() {
            common::save_distribution_chart(
                &figures.join("embed_distribution.png"),
                "ZUNA — Embedding value distribution  (ideal: N(0,1) via MMD)",
                &all_vals, 60,
            ).unwrap_or_else(|e| eprintln!("⚠  distribution chart: {e}"));
        }

        if !dim_means.is_empty() {
            common::save_dim_stats_chart(
                &figures.join("embed_dim_stats.png"),
                "ZUNA — Per-dimension embedding statistics  (mean ± std across tokens)",
                &dim_means, &dim_stds,
            ).unwrap_or_else(|e| eprintln!("⚠  dim-stats chart: {e}"));
        }

        timer.done(&format!("charts → {}/", args.figures));
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let ms_total = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("\n── Summary ────────────────────────────────────────────────");
    println!("  Weights  : {ms_weights:.0} ms");
    println!("  Preproc  : {ms_preproc:.1} ms");
    println!("  Encode   : {ms_encode:.1} ms  ({n_epochs} epochs)");
    println!("  Total    : {ms_total:.0} ms");
    println!("  Output   : {}", args.output);
    if let Some(ep) = result.epochs.first() {
        println!("  Emb dim  : {} × {} = {} values/epoch",
            ep.n_tokens(), ep.output_dim(), ep.embeddings.len());
    }
    eprintln!("TIMING weights={ms_weights:.1}ms preproc={ms_preproc:.1}ms \
               encode={ms_encode:.1}ms total={ms_total:.1}ms");
    Ok(())
}

// ── Helper: export encoder inputs to safetensors ─────────────────────────────

fn export_encoder_inputs<B: Backend>(
    batches: &[InputBatch<B>],
    path:    &str,
) -> anyhow::Result<()> {
    use safetensors::{Dtype, View};
    use std::borrow::Cow;

    struct RawT { data: Vec<u8>, shape: Vec<usize>, dt: Dtype }
    impl View for RawT {
        fn dtype(&self)    -> Dtype         { self.dt }
        fn shape(&self)    -> &[usize]      { &self.shape }
        fn data(&self)     -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
        fn data_len(&self) -> usize          { self.data.len() }
    }

    let f32_b = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let i64_b = |v: &[i64]| v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>();

    let mut keys: Vec<String> = Vec::new();
    let mut tensors: Vec<RawT> = Vec::new();

    for (i, batch) in batches.iter().enumerate() {
        let [_, s, d] = batch.encoder_input.dims();

        // encoder_input: [1, S, 32] → flatten to [S, 32]
        let inp_vec = batch.encoder_input.clone().reshape([s, d])
            .into_data().to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("encoder_input→vec: {e:?}"))?;
        keys.push(format!("encoder_input_{i}"));
        tensors.push(RawT { data: f32_b(&inp_vec), shape: vec![s, d], dt: Dtype::F32 });

        // tok_idx: [S, 4]
        let [s2, c] = batch.tok_idx.dims();
        // NdArray stores Int as i64; wgpu stores Int as i32 — fall back and widen.
        let idx_data = batch.tok_idx.clone().into_data();
        let idx_vec: Vec<i64> = idx_data.to_vec::<i64>()
            .or_else(|_| idx_data.to_vec::<i32>()
                .map(|v| v.into_iter().map(|x| x as i64).collect()))
            .map_err(|e| anyhow::anyhow!("tok_idx→vec: {e:?}"))?;
        keys.push(format!("tok_idx_{i}"));
        tensors.push(RawT { data: i64_b(&idx_vec), shape: vec![s2, c], dt: Dtype::I64 });
    }

    let n = batches.len() as f32;
    keys.push("n_epochs".into());
    tensors.push(RawT { data: f32_b(&[n]), shape: vec![1], dt: Dtype::F32 });

    let pairs: Vec<(&str, RawT)> = keys.iter().map(|k| k.as_str()).zip(tensors).collect();
    let bytes = safetensors::serialize(pairs, None)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

// ── Helper: per-dimension embedding statistics ────────────────────────────────

fn compute_dim_stats(
    epochs: &[zuna_rs::EpochEmbedding],
    n_dims: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut sums   = vec![0.0f64; n_dims];
    let mut sq_sums = vec![0.0f64; n_dims];
    let mut counts = vec![0usize; n_dims];
    for ep in epochs {
        for (i, &v) in ep.embeddings.iter().enumerate() {
            let d = i % n_dims;
            sums[d]    += v as f64;
            sq_sums[d] += (v * v) as f64;
            counts[d]  += 1;
        }
    }
    let means: Vec<f64> = sums.iter().zip(&counts).map(|(&s, &n)| s / n as f64).collect();
    let stds:  Vec<f64> = sq_sums.iter().zip(&sums).zip(&counts)
        .map(|((&sq, &s), &n)| ((sq / n as f64) - (s / n as f64).powi(2)).max(0.0).sqrt())
        .collect();
    (means, stds)
}
