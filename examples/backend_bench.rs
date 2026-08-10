//! ZUNA backend × channel-count benchmark.
//!
//! Sweeps the encoder forward pass across every compiled-in backend and a range
//! of channel counts.  Produces a CSV-style table on stdout and (optionally) a
//! heatmap chart.
//!
//! # Usage
//!
//! ```sh
//! # Build with all backends you want to compare:
//! cargo build --release --features ndarray,mlx,wgpu --example backend_bench
//!
//! # Run (will test every compiled-in backend):
//! cargo run --example backend_bench --release --features ndarray,mlx,wgpu
//!
//! # Custom channel counts and warmup:
//! cargo run --example backend_bench --release --features ndarray,mlx -- \
//!     --channels 4,8,12,19,32 --runs 3 --warmup 1
//! ```

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Distribution;
use clap::Parser;
use zuna_rs::{config::DataConfig, data::InputBatch, ZunaEncoder};

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "backend_bench",
    about = "ZUNA — sweep encoder across backends × channel counts"
)]
struct Args {
    /// Comma-separated list of channel counts to benchmark.
    #[arg(long, default_value = "4,8,12,19,32,64")]
    channels: String,

    /// Number of timed runs per (backend, channel_count) pair.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Warmup runs (not timed) to prime GPU caches.
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of CPU threads for NdArray backend.
    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    /// HuggingFace repo ID for automatic weight resolution.
    #[arg(long, default_value = common::DEFAULT_REPO, env = "ZUNA_REPO")]
    repo: String,

    #[arg(long, env = "HF_HOME")]
    hf_cache: Option<std::path::PathBuf>,

    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<String>,

    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<String>,

    /// Directory for output chart.
    #[arg(long, default_value = "figures")]
    figures: String,

    /// Skip chart generation.
    #[arg(long)]
    no_charts: bool,
}

// ── Synthetic batch generation ───────────────────────────────────────────────

/// Create a synthetic InputBatch with `n_ch` channels.
///
/// Uses random Normal(0,1) data and evenly-spaced channel positions,
/// mimicking a real EEG epoch at 256 Hz × 5 s = 1280 samples.
fn make_synthetic_batch<B: Backend>(n_ch: usize, device: &B::Device) -> InputBatch<B> {
    let cfg = DataConfig::default();
    let tf = cfg.num_fine_time_pts; // 32
    let epoch_samples = 1280usize; // 5s × 256Hz
    let tc = epoch_samples / tf; // 40
    let n_tokens = n_ch * tc;

    // Random encoder input [1, n_tokens, 32]
    let encoder_input =
        Tensor::<B, 3>::random([1, n_tokens, tf], Distribution::Normal(0.0, 1.0), device);

    // Build tok_idx [n_tokens, 4]: (x_bin, y_bin, z_bin, t_coarse)
    let mut idx_data = vec![0i64; n_tokens * 4];
    for ch in 0..n_ch {
        let x_bin = (ch * 49 / n_ch.max(1)) as i64;
        let y_bin = ((ch * 7) % 50) as i64;
        let z_bin = 25i64;
        for t in 0..tc {
            let row = ch * tc + t;
            idx_data[row * 4] = x_bin;
            idx_data[row * 4 + 1] = y_bin;
            idx_data[row * 4 + 2] = z_bin;
            idx_data[row * 4 + 3] = t as i64;
        }
    }
    let tok_idx = Tensor::<B, 2, Int>::from_data(TensorData::new(idx_data, [n_tokens, 4]), device);

    // Dummy channel positions [n_ch, 3]
    let chan_pos = Tensor::<B, 2>::random([n_ch, 3], Distribution::Normal(0.0, 0.05), device);

    InputBatch {
        encoder_input,
        tok_idx,
        chan_pos,
        n_channels: n_ch,
        tc,
    }
}

// ── Benchmark runner ─────────────────────────────────────────────────────────

struct BenchResult {
    backend: String,
    n_channels: usize,
    n_tokens: usize,
    runs: Vec<f64>,
}

impl BenchResult {
    fn mean_ms(&self) -> f64 {
        self.runs.iter().sum::<f64>() / self.runs.len() as f64
    }
    fn std_ms(&self) -> f64 {
        let m = self.mean_ms();
        (self.runs.iter().map(|v| (v - m).powi(2)).sum::<f64>() / self.runs.len() as f64).sqrt()
    }
    fn min_ms(&self) -> f64 {
        self.runs.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn per_epoch_ms(&self) -> f64 {
        self.min_ms() // single epoch (we benchmark 1 epoch at a time)
    }
}

fn bench_backend<B: Backend>(
    backend_name: &str,
    encoder: &ZunaEncoder<B>,
    channel_counts: &[usize],
    n_warmup: usize,
    n_runs: usize,
) -> Vec<BenchResult> {
    let device = encoder.device();
    let mut results = Vec::new();

    for &n_ch in channel_counts {
        let batch = make_synthetic_batch::<B>(n_ch, device);
        let n_tokens = n_ch * batch.tc;

        // Warmup
        for _ in 0..n_warmup {
            let b = make_synthetic_batch::<B>(n_ch, device);
            let _ = encoder.encode_batches(vec![b]);
        }

        // Timed runs
        let mut runs = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let b = make_synthetic_batch::<B>(n_ch, device);
            let t = Instant::now();
            let _ = encoder.encode_batches(vec![b]);
            runs.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        eprint!(".");
        results.push(BenchResult {
            backend: backend_name.to_string(),
            n_channels: n_ch,
            n_tokens,
            runs,
        });
    }
    eprintln!();

    results
}

// ── Per-backend dispatch ─────────────────────────────────────────────────────

fn run_all(
    weights_path: &Path,
    config_path: &Path,
    channel_counts: &[usize],
    n_warmup: usize,
    n_runs: usize,
) -> Vec<BenchResult> {
    let mut all_results = Vec::new();

    #[cfg(feature = "ndarray")]
    {
        use burn::backend::{ndarray::NdArrayDevice, NdArray};
        let name = if cfg!(feature = "blas-accelerate") {
            "NdArray+Accelerate"
        } else if cfg!(feature = "openblas-system") {
            "NdArray+OpenBLAS"
        } else {
            "NdArray+Rayon"
        };
        eprint!("  {name:<22} ");
        match ZunaEncoder::<NdArray>::load(config_path, weights_path, NdArrayDevice::Cpu) {
            Ok((enc, _)) => {
                all_results.extend(bench_backend(name, &enc, channel_counts, n_warmup, n_runs));
            }
            Err(e) => eprintln!("SKIP ({e})"),
        }
    }

    #[cfg(any(feature = "wgpu", feature = "wgpu-f16"))]
    {
        use burn::backend::{wgpu::WgpuDevice, Wgpu};
        let name = "wgpu f32";
        eprint!("  {name:<22} ");
        match ZunaEncoder::<Wgpu>::load(config_path, weights_path, WgpuDevice::DefaultDevice) {
            Ok((enc, _)) => {
                all_results.extend(bench_backend(name, &enc, channel_counts, n_warmup, n_runs));
            }
            Err(e) => eprintln!("SKIP ({e})"),
        }
    }

    #[cfg(any(feature = "wgpu-f16", feature = "wgpu"))]
    {
        type B = burn::backend::wgpu::Wgpu<half::f16, i32, u32>;
        let name = "wgpu f16";
        eprint!("  {name:<22} ");
        match ZunaEncoder::<B>::load(
            config_path,
            weights_path,
            burn::backend::wgpu::WgpuDevice::DefaultDevice,
        ) {
            Ok((enc, _)) => {
                all_results.extend(bench_backend(name, &enc, channel_counts, n_warmup, n_runs));
            }
            Err(e) => eprintln!("SKIP ({e})"),
        }
    }

    #[cfg(any(feature = "mlx", feature = "mlx-f16"))]
    {
        use burn_mlx::{Mlx, MlxDevice};
        let name = "MLX f32";
        eprint!("  {name:<22} ");
        match ZunaEncoder::<Mlx>::load(config_path, weights_path, MlxDevice::Gpu) {
            Ok((enc, _)) => {
                all_results.extend(bench_backend(name, &enc, channel_counts, n_warmup, n_runs));
            }
            Err(e) => eprintln!("SKIP ({e})"),
        }
    }

    #[cfg(any(feature = "mlx-f16", feature = "mlx"))]
    {
        use burn_mlx::{MlxDevice, MlxHalf};
        let name = "MLX f16";
        eprint!("  {name:<22} ");
        match ZunaEncoder::<MlxHalf>::load(config_path, weights_path, MlxDevice::Gpu) {
            Ok((enc, _)) => {
                all_results.extend(bench_backend(name, &enc, channel_counts, n_warmup, n_runs));
            }
            Err(e) => eprintln!("SKIP ({e})"),
        }
    }

    all_results
}

// ── Chart generation ─────────────────────────────────────────────────────────

fn generate_chart(
    results: &[BenchResult],
    channel_counts: &[usize],
    out_path: &Path,
) -> anyhow::Result<()> {
    use plotters::prelude::*;

    // Group by backend
    let backends: Vec<String> = {
        let mut seen = Vec::new();
        for r in results {
            if !seen.contains(&r.backend) {
                seen.push(r.backend.clone());
            }
        }
        seen
    };

    let max_ms = results.iter().map(|r| r.min_ms()).fold(0.0f64, f64::max) * 1.15;

    let root = BitMapBackend::new(out_path, (1000, 550)).into_drawing_area();
    root.fill(&WHITE)?;

    let n_backends = backends.len();
    let n_channels = channel_counts.len();
    let total_groups = n_channels;
    let bar_width = 0.7 / n_backends as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "ZUNA Encoder — Backend × Channel Count",
            ("sans-serif", 20).into_font(),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(70)
        .build_cartesian_2d(0f64..(total_groups as f64), 0f64..max_ms)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Encode time (ms, 1 epoch, best of N)")
        .x_desc("Channels")
        .x_labels(n_channels)
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < channel_counts.len() {
                format!("{}", channel_counts[idx])
            } else {
                String::new()
            }
        })
        .draw()?;

    let colors = [
        RGBColor(108, 117, 125), // grey (NdArray)
        RGBColor(13, 110, 253),  // blue (wgpu f32)
        RGBColor(10, 88, 202),   // dark blue (wgpu f16)
        RGBColor(25, 135, 84),   // green (MLX f32)
        RGBColor(15, 81, 50),    // dark green (MLX f16)
    ];

    for (bi, backend) in backends.iter().enumerate() {
        let color = colors[bi % colors.len()];
        let backend_results: Vec<&BenchResult> =
            results.iter().filter(|r| &r.backend == backend).collect();

        let data: Vec<(f64, f64)> = backend_results
            .iter()
            .enumerate()
            .map(|(ci, r)| {
                let x = ci as f64 + (bi as f64 - (n_backends as f64 - 1.0) / 2.0) * bar_width;
                (x, r.min_ms())
            })
            .collect();

        chart
            .draw_series(data.iter().map(|&(x, y)| {
                let x0 = x - bar_width * 0.4;
                let x1 = x + bar_width * 0.4;
                Rectangle::new([(x0, 0.0), (x1, y)], color.filled())
            }))?
            .label(backend.as_str())
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 15, y + 5)], color.filled()));

        // Value labels
        for &(x, y) in &data {
            let label = format!("{y:.0}");
            chart.draw_series(std::iter::once(Text::new(
                label,
                (x, y + max_ms * 0.02),
                ("sans-serif", 9).into_font(),
            )))?;
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK.mix(0.3))
        .background_style(WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _n = zuna_rs::init_threads(args.threads);

    let channel_counts: Vec<usize> = args
        .channels
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("invalid channel count"))
        .collect();

    println!("=== ZUNA Backend × Channel Benchmark ===");
    println!("  channels : {:?}", channel_counts);
    println!("  runs     : {} (+ {} warmup)", args.runs, args.warmup);
    println!();

    // Resolve weights
    let (weights_path, config_path) = common::resolve_weights(
        &args.repo,
        args.weights.as_deref(),
        args.config.as_deref(),
        args.hf_cache.as_deref(),
    )?;
    println!("  weights  : {}", weights_path.display());
    println!();

    println!("Loading and benchmarking backends...");
    let results = run_all(
        &weights_path,
        &config_path,
        &channel_counts,
        args.warmup,
        args.runs,
    );

    // Print results table
    println!();
    println!(
        "{:<22} {:>4} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "Backend", "Ch", "Tok", "Min(ms)", "Mean(ms)", "Std(ms)", "ms/epoch"
    );
    println!("{}", "─".repeat(78));

    for r in &results {
        println!(
            "{:<22} {:>4} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            r.backend,
            r.n_channels,
            r.n_tokens,
            r.min_ms(),
            r.mean_ms(),
            r.std_ms(),
            r.per_epoch_ms()
        );
    }

    // Print speedup table (vs slowest backend for each channel count)
    println!();
    println!("── Speedup vs NdArray baseline ─────────────────────────────────");
    let backends: Vec<String> = {
        let mut seen = Vec::new();
        for r in &results {
            if !seen.contains(&r.backend) {
                seen.push(r.backend.clone());
            }
        }
        seen
    };

    // Header
    print!("{:<22}", "Backend");
    for &ch in &channel_counts {
        print!(" {:>6}ch", ch);
    }
    println!();
    println!("{}", "─".repeat(22 + channel_counts.len() * 8));

    for backend in &backends {
        print!("{:<22}", backend);
        for &ch in &channel_counts {
            let this = results
                .iter()
                .find(|r| r.backend == *backend && r.n_channels == ch)
                .map(|r| r.min_ms());
            let baseline = results
                .iter()
                .find(|r| r.n_channels == ch)
                .map(|r| r.min_ms());
            match (this, baseline) {
                (Some(t), Some(b)) => print!(" {:>6.1}x", b / t),
                _ => print!(" {:>7}", "-"),
            }
        }
        println!();
    }

    // Generate chart
    if !args.no_charts {
        let figures = std::path::PathBuf::from(&args.figures);
        common::ensure_figures_dir(&figures)?;
        let chart_path = figures.join("backend_channel_bench.png");
        match generate_chart(&results, &channel_counts, &chart_path) {
            Ok(()) => println!("\nChart → {}", chart_path.display()),
            Err(e) => eprintln!("\nChart error: {e}"),
        }
    }

    // Also output as CSV for easy ingestion
    println!("\n── CSV ─────────────────────────────────────────────────────────");
    println!("backend,channels,tokens,min_ms,mean_ms,std_ms");
    for r in &results {
        println!(
            "{},{},{},{:.1},{:.1},{:.1}",
            r.backend,
            r.n_channels,
            r.n_tokens,
            r.min_ms(),
            r.mean_ms(),
            r.std_ms()
        );
    }

    Ok(())
}
