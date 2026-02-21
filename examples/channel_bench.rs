//! Comprehensive benchmark: EEG channel-padding strategies vs. UMAP embedding space.
//!
//! Tests 26 configurations spanning 6 channel counts × 5 padding strategies
//! (+ 1 full-channel baseline), runs the ZUNA encoder on each, collects the
//! resulting embedding vectors, then spawns a Python UMAP script that produces
//! 2-D and 3-D scatter plots coloured by configuration.
//!
//! # Usage
//!
//! ```sh
//! # Basic run (CPU, auto-download weights)
//! cargo run --example channel_bench --release -- --features hf-download
//!
//! # Custom FIF + explicit weights + GPU
//! cargo run --example channel_bench --release \
//!     --no-default-features --features wgpu -- \
//!     --device gpu \
//!     --fif data/sample1_raw.fif \
//!     --weights model.safetensors --config config.json
//!
//! # Skip UMAP (just produce JSON + NPY)
//! cargo run --example channel_bench --release -- --skip-umap
//! ```
//!
//! ## Configurations benchmarked
//!
//! | Count | Channels kept              | Strategies (×5)                          |
//! |-------|----------------------------|------------------------------------------|
//! | 12    | all (baseline)             | –                                         |
//! | 10    | drop F7, F8                | zero / nearest / clone-fp1 / noisy / xyz |
//! | 8     | drop F7 F8 O1 O2           | "                                         |
//! | 6     | Fp1 Fp2 C3 C4 P3 P4        | "                                         |
//! | 4     | Fp1 Fp2 C3 C4              | "                                         |
//! | 2     | Fp1 Fp2                    | "                                         |
//!
//! ## Output files  (all in --output-dir, default `figures/`)
//!
//! | File                              | Contents                              |
//! |-----------------------------------|---------------------------------------|
//! | `channel_bench_results.json`      | All configs + per-epoch stats + timing|
//! | `channel_bench_embeddings.npy`    | [N, 32] f32 — all embedding vectors   |
//! | `channel_bench_labels.npy`        | [N] i32  — config index per vector    |
//! | `umap_2d.png`                     | 2-D UMAP / t-SNE / PCA scatter        |
//! | `umap_3d.png`                     | 3-D scatter (4 orthographic panels)   |

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::Backend;
use clap::{Parser, ValueEnum};
use ndarray::Array2;
use zuna_rs::{
    channel_positions::{channel_xyz, nearest_channel},
    config::DataConfig,
    load_from_raw_tensor, ZunaEncoder,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants — target channel layout (always 12 channels in encoder)
// ─────────────────────────────────────────────────────────────────────────────

/// The 12 channels from the sample FIF file (used as the target layout).
const ALL_12: &[&str] = &[
    "Fp1", "Fp2", "F3", "F4", "C3", "C4",
    "P3",  "P4",  "O1", "O2", "F7", "F8",
];

/// Which channels to keep for each channel-count tier.
const SUBSETS: &[(&str, &[&str])] = &[
    ("12ch", &["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8"]),
    ("10ch", &["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2"]),
    ("8ch",  &["Fp1","Fp2","F3","F4","C3","C4","P3","P4"]),
    ("6ch",  &["Fp1","Fp2","C3","C4","P3","P4"]),
    ("4ch",  &["Fp1","Fp2","C3","C4"]),
    ("2ch",  &["Fp1","Fp2"]),
];

/// Padding strategies tested for each sub-12 subset.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Strategy {
    // ── original five ────────────────────────────────────────────────────────
    /// All-zeros signal; position from montage database.
    ZeroPad,
    /// Clone data from the nearest kept channel (by database XYZ distance).
    CloneNearest,
    /// Clone data from Fp1 for every missing channel.
    CloneFp1,
    /// Clone nearest + Gaussian noise proportional to source RMS.
    NoisyClone,
    /// Same data as CloneNearest; XYZ of missing channels jittered by ±5 mm.
    XyzJitter,
    // ── new strategies ───────────────────────────────────────────────────────
    /// Inverse-distance–weighted average of the 3 nearest kept channels.
    InterpWeighted,
    /// Nearest kept channel on the opposite hemisphere (flip X, find nearest).
    Mirror,
    /// Per-sample mean of all kept channels (common-average-reference signal).
    MeanRef,
    /// **No padding** — encode only the channels that are actually present.
    /// The encoder receives a shorter token sequence; no data is invented.
    Native,
}

impl Strategy {
    fn label(self) -> &'static str {
        match self {
            Self::ZeroPad        => "zero_pad",
            Self::CloneNearest   => "clone_nearest",
            Self::CloneFp1       => "clone_fp1",
            Self::NoisyClone     => "noisy_clone",
            Self::XyzJitter      => "xyz_jitter",
            Self::InterpWeighted => "interp_weighted",
            Self::Mirror         => "mirror",
            Self::MeanRef        => "mean_ref",
            Self::Native         => "native",
        }
    }
    fn all() -> &'static [Strategy] {
        &[
            Self::ZeroPad, Self::CloneNearest, Self::CloneFp1,
            Self::NoisyClone, Self::XyzJitter,
            Self::InterpWeighted, Self::Mirror, Self::MeanRef,
            Self::Native,
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark mode
// ─────────────────────────────────────────────────────────────────────────────

/// Which set of configurations to run.
#[derive(Debug, Clone, ValueEnum, PartialEq)]
enum Mode {
    /// All sections: strategy comparison + channel ablation + region dropout.
    All,
    /// Original 5 strategies + 3 new strategies across 5 channel counts (41 configs).
    Strategy,
    /// Drop each of the 12 channels one at a time using CloneNearest (12 configs).
    Ablation,
    /// Anatomical region dropout: frontal / central / posterior / left / right (10 configs).
    Region,
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum DeviceArg { Cpu, Gpu }

#[derive(Parser, Debug)]
#[command(
    name  = "channel_bench",
    about = "Benchmark EEG channel-padding strategies and visualise in UMAP space",
)]
struct Args {
    /// Source FIF recording.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif"))]
    fif: String,

    /// Directory for figures and JSON results.
    #[arg(long, default_value = "figures")]
    output_dir: String,

    /// Directory for NPY data files (embeddings + labels).
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data"))]
    data_dir: String,

    /// Which configurations to run.
    #[arg(long, default_value = "strategy")]
    mode: Mode,

    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: DeviceArg,

    /// Signal normalisation divisor.
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Noise factor for NoisyClone  (noise_std = factor × channel_rms).
    #[arg(long, default_value_t = 0.20)]
    noise_factor: f32,

    /// Position jitter for XyzJitter strategy (metres, 1σ Gaussian).
    #[arg(long, default_value_t = 0.005)]
    xyz_jitter_m: f32,

    /// Maximum embedding vectors fed to UMAP/t-SNE (subsampled if exceeded).
    #[arg(long, default_value_t = 30_000)]
    max_umap_pts: usize,

    /// Skip the Python UMAP step (just produce JSON + NPY).
    #[arg(long)]
    skip_umap: bool,

    /// HuggingFace model repo ID.
    #[arg(long, default_value = "Zyphra/ZUNA", env = "ZUNA_REPO")]
    repo: String,

    /// Explicit safetensors weights path.
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<PathBuf>,

    /// Explicit config.json path (must pair with --weights).
    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<PathBuf>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark configuration spec
// ─────────────────────────────────────────────────────────────────────────────

/// One fully-described benchmark run (what channels to keep, what strategy).
#[derive(Debug, Clone)]
struct BenchSpec {
    /// Human-readable label written to JSON / table.
    label:    String,
    /// Which of the 12 FIF channels are treated as real (others are padded).
    kept:     Vec<&'static str>,
    /// How to synthesise missing channels.
    strategy: Strategy,
    /// Section label for grouping in JSON output.
    section:  &'static str,
}

/// Build the full list of configs for the requested mode.
fn build_configs(mode: &Mode) -> Vec<BenchSpec> {
    let mut specs: Vec<BenchSpec> = Vec::new();

    // ── Strategy section ─────────────────────────────────────────────────────
    if matches!(mode, Mode::Strategy | Mode::All) {
        for &(count_label, kept) in SUBSETS {
            let strategies: &[Strategy] = if kept.len() == 12 {
                &[Strategy::ZeroPad]   // baseline — padding does nothing
            } else {
                Strategy::all()
            };
            for &strat in strategies {
                let label = if kept.len() == 12 {
                    "12ch_baseline".to_string()
                } else {
                    format!("{}_{}", count_label, strat.label())
                };
                specs.push(BenchSpec {
                    label,
                    kept: kept.to_vec(),
                    strategy: strat,
                    section:  "strategy",
                });
            }
        }
    }

    // ── Ablation section ─────────────────────────────────────────────────────
    // Drop each of the 12 FIF channels in turn; fill it with CloneNearest.
    if matches!(mode, Mode::Ablation | Mode::All) {
        for &dropped in ALL_12 {
            let kept: Vec<&'static str> = ALL_12.iter()
                .copied()
                .filter(|&n| !n.eq_ignore_ascii_case(dropped))
                .collect();
            specs.push(BenchSpec {
                label:   format!("drop_{dropped}"),
                kept,
                strategy: Strategy::CloneNearest,
                section:  "ablation",
            });
        }
    }

    // ── Region section ───────────────────────────────────────────────────────
    // Anatomical sub-montages: drop a whole scalp region.
    if matches!(mode, Mode::Region | Mode::All) {
        /// (label, channels to DROP, strategies to test)
        const REGIONS: &[(&str, &[&str], &[Strategy])] = &[
            // Frontal electrodes — Fp1/2 + lateral frontal F7/8
            ("frontal_drop",   &["Fp1","Fp2","F7","F8"],
             &[Strategy::ZeroPad, Strategy::CloneNearest, Strategy::Mirror, Strategy::MeanRef]),
            // Mid-frontal + central motor strip
            ("central_drop",   &["F3","F4","C3","C4"],
             &[Strategy::ZeroPad, Strategy::CloneNearest, Strategy::Mirror, Strategy::MeanRef]),
            // Parieto-occipital
            ("posterior_drop", &["P3","P4","O1","O2"],
             &[Strategy::ZeroPad, Strategy::CloneNearest, Strategy::Mirror, Strategy::MeanRef]),
            // Full left hemisphere
            ("left_hemi_drop", &["Fp1","F3","C3","P3","O1","F7"],
             &[Strategy::ZeroPad, Strategy::CloneNearest, Strategy::Mirror, Strategy::MeanRef]),
            // Full right hemisphere
            ("right_hemi_drop",&["Fp2","F4","C4","P4","O2","F8"],
             &[Strategy::ZeroPad, Strategy::CloneNearest, Strategy::Mirror, Strategy::MeanRef]),
        ];
        for &(region_label, dropped, strats) in REGIONS {
            let kept: Vec<&'static str> = ALL_12.iter()
                .copied()
                .filter(|&n| !dropped.iter().any(|d| d.eq_ignore_ascii_case(n)))
                .collect();
            for &strat in strats {
                specs.push(BenchSpec {
                    label:   format!("{}_{}", region_label, strat.label()),
                    kept:    kept.clone(),
                    strategy: strat,
                    section:  "region",
                });
            }
        }
    }

    specs
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.device {
        DeviceArg::Cpu => run_cpu(args),
        DeviceArg::Gpu => run_gpu(args),
    }
}

#[cfg(feature = "ndarray")]
fn run_cpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    run::<NdArray>(NdArrayDevice::Cpu, args)
}
#[cfg(not(feature = "ndarray"))]
fn run_cpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("rebuild with --features ndarray")
}

#[cfg(feature = "wgpu")]
fn run_gpu(args: Args) -> anyhow::Result<()> {
    use burn::backend::{wgpu::WgpuDevice, Wgpu};
    run::<Wgpu>(WgpuDevice::DefaultDevice, args)
}
#[cfg(not(feature = "wgpu"))]
fn run_gpu(_: Args) -> anyhow::Result<()> {
    anyhow::bail!("rebuild with --no-default-features --features wgpu")
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark runner
// ─────────────────────────────────────────────────────────────────────────────

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()>
where
    B::Device: Clone,
{
    let t_total = Instant::now();
    let out_dir = PathBuf::from(&args.output_dir);
    std::fs::create_dir_all(&out_dir)?;

    // ── 1. Resolve weights ────────────────────────────────────────────────────
    println!("=== ZUNA Channel Benchmark ===\n");
    print!("[1/5] Resolving weights ... ");
    let (weights_path, config_path) =
        resolve_weights(&args.repo, args.weights.clone(), args.config.clone())?;
    println!("OK");

    // ── 2. Load encoder once ──────────────────────────────────────────────────
    print!("[2/5] Loading encoder ... ");
    let t_enc_load = Instant::now();
    let data_device = device.clone();
    let (enc, _) = ZunaEncoder::<B>::load(&config_path, &weights_path, device)?;
    println!("OK ({:.0} ms)  {}", t_enc_load.elapsed().as_millis(), enc.describe());

    // ── 3. Read raw FIF data ──────────────────────────────────────────────────
    print!("[3/5] Reading FIF data: {} ... ", args.fif);
    let (raw_data, src_sfreq) = read_raw_fif(Path::new(&args.fif))?;
    println!("OK  shape={:?}  sfreq={:.0} Hz", raw_data.dim(), src_sfreq);

    // ── 4. Run all benchmark configs ──────────────────────────────────────────
    let configs = build_configs(&args.mode);
    println!("[4/5] Running {} configurations  (mode: {:?}) ...", configs.len(), args.mode);
    println!("{:<34} {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "config", "epochs", "mean", "std", "min", "max", "enc_ms");
    println!("{}", "─".repeat(84));

    let mut prng = Prng::new(42);
    let mut all_embeddings: Vec<f32> = Vec::new();
    let mut all_labels: Vec<i32>     = Vec::new();
    let mut config_records: Vec<serde_json::Value> = Vec::new();
    let mut config_names: Vec<String> = Vec::new();

    for (config_idx, spec) in configs.iter().enumerate() {
        let t_cfg = Instant::now();

        // Build padded [12, T] data + positions for this config
        let (padded, positions) = build_padded_config(
            &raw_data,
            &spec.kept,
            spec.strategy,
            args.noise_factor,
            args.xyz_jitter_m,
            &mut prng,
        );

        // Run preprocessing pipeline + encode
        let batches = load_from_raw_tensor::<B>(
            padded,
            &positions,
            src_sfreq,
            args.data_norm,
            &DataConfig::default(),
            &data_device,
        )?;
        let n_epochs = batches.len();
        let n_tokens = batches.first().map(|b| b.n_channels * b.tc).unwrap_or(0);

        let t_enc_start = Instant::now();
        let epoch_embs = enc.encode_batches(batches)?;
        let enc_ms = t_enc_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t_cfg.elapsed().as_secs_f64() * 1000.0;

        // Collect embedding vectors
        let mut all_vals: Vec<f32> = Vec::new();
        for ep in &epoch_embs {
            all_vals.extend_from_slice(&ep.embeddings);
        }

        // Stats
        let n = all_vals.len() as f64;
        let mean = all_vals.iter().map(|&v| v as f64).sum::<f64>() / n;
        let std  = (all_vals.iter().map(|&v| { let d = v as f64 - mean; d*d }).sum::<f64>() / n).sqrt();
        let min  = all_vals.iter().cloned().fold( f32::INFINITY, f32::min);
        let max  = all_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_abs = all_vals.iter().map(|v| v.abs()).sum::<f32>() / all_vals.len() as f32;

        // Collect for UMAP
        let n_vecs = all_vals.len() / 32;   // output_dim = 32
        all_embeddings.extend_from_slice(&all_vals);
        for _ in 0..n_vecs { all_labels.push(config_idx as i32); }

        println!("{:<34} {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.1}",
            spec.label, n_epochs, mean, std, min, max, enc_ms);

        config_records.push(serde_json::json!({
            "index":              config_idx,
            "name":               spec.label,
            "section":            spec.section,
            "n_channels_loaded":  spec.kept.len(),
            "n_channels_target":  ALL_12.len(),
            "n_channels_padded":  ALL_12.len() - spec.kept.len(),
            "strategy":           spec.strategy.label(),
            "channels_loaded":    spec.kept,
            "channels_target":    ALL_12,
            "noise_factor":       if matches!(spec.strategy, Strategy::NoisyClone) { args.noise_factor } else { 0.0 },
            "xyz_jitter_m":       if matches!(spec.strategy, Strategy::XyzJitter)  { args.xyz_jitter_m } else { 0.0 },
            "n_epochs":           n_epochs,
            "n_tokens_per_epoch": n_tokens,
            "embedding_dim":      32,
            "n_total_vectors":    n_vecs,
            "stats": { "mean": mean, "std": std, "min": min, "max": max, "mean_abs": mean_abs },
            "timing_ms": { "total": total_ms, "encode": enc_ms },
        }));
        config_names.push(spec.label.clone());
    }

    let total_vectors = all_embeddings.len() / 32;
    println!("{}", "─".repeat(84));
    println!("Total: {} configs  {}×32 embedding vectors  {:.1}s",
        configs.len(), total_vectors, t_total.elapsed().as_secs_f64());

    // ── 5. Save results ───────────────────────────────────────────────────────
    println!("\n[5/5] Saving results ...");
    let data_dir = PathBuf::from(&args.data_dir);
    std::fs::create_dir_all(&data_dir)?;

    // 5a. NPY files go in data/
    let mode_tag = format!("{:?}", args.mode).to_lowercase();
    let emb_npy  = data_dir.join(format!("channel_bench_{mode_tag}_embeddings.npy"));
    let lbl_npy  = data_dir.join(format!("channel_bench_{mode_tag}_labels.npy"));
    write_npy_f32_2d(&emb_npy, &all_embeddings, total_vectors, 32)?;
    write_npy_i32_1d(&lbl_npy, &all_labels)?;
    println!("  embeddings   → {} ({} × 32)", emb_npy.display(), total_vectors);
    println!("  labels       → {}", lbl_npy.display());

    // 5b. JSON results go in figures/
    let json_path = out_dir.join(format!("channel_bench_{mode_tag}_results.json"));
    let results = serde_json::json!({
        "timestamp":      chrono_now(),
        "fif_file":       args.fif,
        "mode":           format!("{:?}", args.mode),
        "n_configs":      configs.len(),
        "total_vectors":  total_vectors,
        "embedding_dim":  32,
        "embeddings_npy": emb_npy.display().to_string(),
        "labels_npy":     lbl_npy.display().to_string(),
        "noise_factor":   args.noise_factor,
        "xyz_jitter_m":   args.xyz_jitter_m,
        "config_names":   config_names,
        "configurations": config_records,
    });
    std::fs::write(&json_path, serde_json::to_string_pretty(&results)?)?;
    println!("  results.json → {}", json_path.display());

    // 5c. Python UMAP step
    if !args.skip_umap {
        println!("\nSpawning Python UMAP visualisation ...");
        let fig_2d = out_dir.join(format!("umap_{mode_tag}_2d.png")).display().to_string();
        let fig_3d = out_dir.join(format!("umap_{mode_tag}_3d.png")).display().to_string();
        run_umap_python(
            &emb_npy.display().to_string(),
            &lbl_npy.display().to_string(),
            &json_path.display().to_string(),
            &fig_2d,
            &fig_3d,
            args.max_umap_pts,
        )?;
        println!("  umap_2d → {fig_2d}");
        println!("  umap_3d → {fig_3d}");
    }

    println!("\nDone.  Total wall-clock: {:.1}s", t_total.elapsed().as_secs_f64());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Data building
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [12, T] padded data array + [12, 3] positions for one benchmark config.
fn build_padded_config(
    raw:          &Array2<f32>,  // [12, T] — full FIF data
    kept:         &[&str],       // channel names to treat as "real"
    strategy:     Strategy,
    noise_factor: f32,
    xyz_jitter_m: f32,
    prng:         &mut Prng,
) -> (Array2<f32>, Vec<[f32; 3]>) {
    // Map channel name → index in ALL_12
    let idx_of = |name: &str| -> usize {
        ALL_12.iter().position(|&n| n.eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("channel '{name}' not in ALL_12"))
    };

    let kept_set: std::collections::HashSet<usize> = kept.iter().map(|n| idx_of(n)).collect();

    // ── Native: return only the kept channels, no synthesis whatsoever ────────
    if strategy == Strategy::Native {
        // Collect rows for kept channels in ALL_12 order
        let mut kept_indices: Vec<usize> = kept_set.iter().copied().collect();
        kept_indices.sort_unstable();                       // deterministic order

        let n_t = raw.ncols();
        let flat: Vec<f32> = kept_indices.iter()
            .flat_map(|&ki| raw.row(ki).to_vec())
            .collect();
        let data = Array2::from_shape_vec((kept_indices.len(), n_t), flat)
            .expect("Native data reshape");

        let positions: Vec<[f32; 3]> = kept_indices.iter()
            .map(|&ki| channel_xyz(ALL_12[ki]).unwrap_or([0.0, 0.0, 0.0]))
            .collect();

        return (data, positions);
    }

    // Positions for all 12 target channels (from montage database)
    let mut positions: Vec<[f32; 3]> = ALL_12.iter()
        .map(|name| channel_xyz(name).unwrap_or([0.0, 0.0, 0.0]))
        .collect();

    // Build output data starting as a copy of the raw 12-channel data
    // (kept channels will be used as-is; missing channels will be overwritten)
    let mut out = raw.clone();

    // Precompute kept channel positions for nearest-neighbour search
    let kept_xyz_idx: Vec<([f32; 3], usize)> = kept.iter()
        .map(|name| {
            let ki = idx_of(name);
            (channel_xyz(name).unwrap_or([0.0, 0.0, 0.0]), ki)
        })
        .collect();

    // Fp1 index for CloneFp1 strategy
    let fp1_idx = idx_of("Fp1");

    for (m_idx, _m_name) in ALL_12.iter().enumerate() {
        if kept_set.contains(&m_idx) { continue; } // real channel — keep as-is

        let m_pos = positions[m_idx]; // database position of missing channel

        match strategy {
            Strategy::ZeroPad => {
                out.row_mut(m_idx).fill(0.0);
                // position stays as database position
            }

            Strategy::CloneNearest => {
                let src_idx = nearest_kept(&kept_xyz_idx, m_pos);
                let src_row: Vec<f32> = raw.row(src_idx).to_vec();
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&src_row));
            }

            Strategy::CloneFp1 => {
                let src_row: Vec<f32> = raw.row(fp1_idx).to_vec();
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&src_row));
            }

            Strategy::NoisyClone => {
                let src_idx = nearest_kept(&kept_xyz_idx, m_pos);
                let src_row: Vec<f32> = raw.row(src_idx).to_vec();
                // RMS of source channel
                let rms = (src_row.iter().map(|&v| (v * v) as f64).sum::<f64>()
                    / src_row.len() as f64).sqrt() as f32;
                let sigma = (noise_factor * rms).max(1e-12) as f64;
                let noisy: Vec<f32> = src_row.iter()
                    .map(|&v| v + prng.normal(sigma))
                    .collect();
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&noisy));
            }

            Strategy::XyzJitter => {
                // Same data as CloneNearest
                let src_idx = nearest_kept(&kept_xyz_idx, m_pos);
                let src_row: Vec<f32> = raw.row(src_idx).to_vec();
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&src_row));
                // Jitter the position of the missing channel
                let sigma = xyz_jitter_m as f64;
                positions[m_idx] = [
                    m_pos[0] + prng.normal(sigma),
                    m_pos[1] + prng.normal(sigma),
                    m_pos[2] + prng.normal(sigma),
                ];
                // Clamp to DataConfig bounds ±0.12 m
                for c in &mut positions[m_idx] {
                    *c = c.clamp(-0.12, 0.12);
                }
            }

            Strategy::InterpWeighted => {
                // Inverse-distance–weighted average of the 3 nearest kept channels.
                const K: usize = 3;
                let mut dists: Vec<(f32, usize)> = kept_xyz_idx.iter()
                    .map(|&(xyz, ki)| (dist3_bench(xyz, m_pos), ki))
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let k_actual = K.min(dists.len()).max(1);
                let k_slice  = &dists[..k_actual];
                let weights: Vec<f32> = k_slice.iter()
                    .map(|(d, _)| if *d < 1e-6 { 1e6_f32 } else { 1.0 / d })
                    .collect();
                let w_sum: f32 = weights.iter().sum();
                let n_t = raw.ncols();
                let mut interp = vec![0f32; n_t];
                for ((_, ki), w) in k_slice.iter().zip(weights.iter()) {
                    let wn = w / w_sum;
                    for (o, &v) in interp.iter_mut().zip(raw.row(*ki).iter()) {
                        *o += wn * v;
                    }
                }
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&interp));
            }

            Strategy::Mirror => {
                // Flip X coordinate to the opposite hemisphere, find nearest kept.
                let mirror_pos = [-m_pos[0], m_pos[1], m_pos[2]];
                let src_idx = nearest_kept(&kept_xyz_idx, mirror_pos);
                let src_row: Vec<f32> = raw.row(src_idx).to_vec();
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&src_row));
            }

            Strategy::MeanRef => {
                // Per-sample mean of all kept channels.
                let n_t   = raw.ncols();
                let n_real = kept_set.len().max(1);
                let mut mean_sig = vec![0f32; n_t];
                for &ki in &kept_set {
                    for (m, &v) in mean_sig.iter_mut().zip(raw.row(ki).iter()) {
                        *m += v;
                    }
                }
                for m in &mut mean_sig { *m /= n_real as f32; }
                out.row_mut(m_idx).assign(&ndarray::ArrayView1::from(&mean_sig));
            }

            // Native returns early above; this arm is unreachable.
            Strategy::Native => unreachable!(),
        }
    }

    (out, positions)
}

/// Euclidean L2 distance between two 3-D points (benchmark-internal helper).
#[inline]
fn dist3_bench(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0]; let dy = a[1] - b[1]; let dz = a[2] - b[2];
    (dx*dx + dy*dy + dz*dz).sqrt()
}

/// Find the index (in ALL_12) of the nearest kept channel to `query_pos`.
fn nearest_kept(kept_xyz_idx: &[([f32; 3], usize)], query_pos: [f32; 3]) -> usize {
    nearest_channel(query_pos, kept_xyz_idx).unwrap_or(kept_xyz_idx[0].1)
}

// ─────────────────────────────────────────────────────────────────────────────
// FIF raw data reader
// ─────────────────────────────────────────────────────────────────────────────

fn read_raw_fif(path: &Path) -> anyhow::Result<(Array2<f32>, f32)> {
    use exg::fiff::raw::open_raw;
    let raw    = open_raw(path)?;
    let sfreq  = raw.info.sfreq as f32;
    let data64 = raw.read_all_data()?;
    // Cast f64 → f32 (matches what load_from_fif does internally)
    let data32 = data64.mapv(|v| v as f32);
    Ok((data32, sfreq))
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal xorshift64 PRNG + Box-Muller
// ─────────────────────────────────────────────────────────────────────────────

struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        // Avoid zero state
        Self(seed ^ 0x9e3779b97f4a7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform f64 in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sample from N(0, sigma²) using Box-Muller.
    fn normal(&mut self, sigma: f64) -> f32 {
        let u1 = self.next_f64().max(1e-15); // avoid log(0)
        let u2 = self.next_f64();
        let z  = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        (z * sigma) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal NPY writer (no external deps)
// ─────────────────────────────────────────────────────────────────────────────

/// Write a 2-D f32 array to a NumPy `.npy` file.
fn write_npy_f32_2d(path: &Path, data: &[f32], nrows: usize, ncols: usize) -> std::io::Result<()> {
    use std::io::Write;
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({nrows}, {ncols}), }}"
    );
    let header_bytes = npy_header_bytes(&dict);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&header_bytes)?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Write a 1-D i32 array to a NumPy `.npy` file.
fn write_npy_i32_1d(path: &Path, data: &[i32]) -> std::io::Result<()> {
    use std::io::Write;
    let dict = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    let header_bytes = npy_header_bytes(&dict);
    let mut f = std::fs::File::create(path)?;
    f.write_all(&header_bytes)?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Build the NPY file header (magic + version + len + padded dict string).
///
/// NumPy format v1.0 spec: the total (10 fixed bytes + HEADER_LEN) must be
/// a multiple of 64.  The header string is space-padded and ends with '\n'.
fn npy_header_bytes(dict: &str) -> Vec<u8> {
    // base_len = dict chars + 1 ('\n' terminator)
    let base_len = dict.len() + 1;
    let fixed    = 10usize; // 6 magic + 1 major + 1 minor + 2 header_len
    // Padding so that (fixed + header_len) % 64 == 0
    let remainder = (fixed + base_len) % 64;
    let n_spaces  = if remainder == 0 { 0 } else { 64 - remainder };
    let header_str = format!("{dict}{}\n", " ".repeat(n_spaces));

    let mut out = Vec::with_capacity(fixed + header_str.len());
    out.extend_from_slice(b"\x93NUMPY"); // magic
    out.push(1u8);                       // major version
    out.push(0u8);                       // minor version
    let hlen = header_str.len() as u16;
    out.extend_from_slice(&hlen.to_le_bytes());
    out.extend_from_slice(header_str.as_bytes());
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Python UMAP visualisation
// ─────────────────────────────────────────────────────────────────────────────

/// UMAP visualisation script — canonical source is `scripts/umap_channel_bench.py`;
/// embedded here via `include_str!` so the binary stays self-contained.
const UMAP_SCRIPT: &str = include_str!("../scripts/umap_channel_bench.py");

fn run_umap_python(
    emb_npy:   &str,
    lbl_npy:   &str,
    json_path: &str,
    out_2d:    &str,
    out_3d:    &str,
    max_pts:   usize,
) -> anyhow::Result<()> {
    // Canonical script location (compiled into the binary path at build time).
    // Falls back to writing the embedded content into /tmp/ if the project
    // directory is not accessible (e.g. binary deployed to another machine).
    let canonical = std::path::PathBuf::from(
        concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/umap_channel_bench.py"),
    );

    let script_path: std::path::PathBuf = if canonical.exists() {
        canonical
    } else {
        // Write embedded copy so the script is always runnable standalone too
        let tmp = std::env::temp_dir().join("zuna_umap_channel_bench.py");
        std::fs::write(&tmp, UMAP_SCRIPT)?;
        eprintln!("  (wrote script to {} — canonical scripts/ dir not found)", tmp.display());
        tmp
    };

    println!("  script: {}", script_path.display());

    let status = std::process::Command::new("python3")
        .args([
            script_path.to_str().unwrap(),
            emb_npy,
            lbl_npy,
            json_path,
            out_2d,
            out_3d,
            &max_pts.to_string(),
        ])
        .status()?;

    if !status.success() {
        eprintln!(
            "⚠  Python UMAP script exited with {:?}.\n\
             JSON + NPY files are intact — run manually:\n\
             \n  python3 {} \\\n    {} \\\n    {} \\\n    {} \\\n    {} \\\n    {} {}",
            status.code(),
            script_path.display(),
            emb_npy, lbl_npy, json_path, out_2d, out_3d, max_pts,
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight resolution (same as csv_embed.rs)
// ─────────────────────────────────────────────────────────────────────────────

fn resolve_weights(
    repo:    &str,
    weights: Option<PathBuf>,
    config:  Option<PathBuf>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    match (weights, config) {
        (Some(w), Some(c)) => return Ok((w, c)),
        (Some(_), None) | (None, Some(_)) =>
            anyhow::bail!("supply both --weights and --config together, or neither"),
        (None, None) => {}
    }

    #[cfg(feature = "hf-download")]
    {
        use hf_hub::api::sync::ApiBuilder;
        let model   = ApiBuilder::new().with_progress(true).build()?.model(repo.to_string());
        let weights = model.get("model-00001-of-00001.safetensors")?;
        let config  = model.get("config.json")?;
        return Ok((weights, config));
    }

    #[allow(unreachable_code)]
    {
        let slug    = repo.replace('/', "--");
        let hf_dir  = hf_cache_dir().join(format!("models--{slug}/snapshots"));
        if hf_dir.is_dir() {
            for entry in std::fs::read_dir(&hf_dir)?.flatten() {
                let snap = entry.path();
                let w = snap.join("model-00001-of-00001.safetensors");
                let c = snap.join("config.json");
                if w.exists() && c.exists() {
                    println!("  (snapshot: {})", snap.file_name().unwrap_or_default().to_string_lossy());
                    return Ok((w, c));
                }
            }
        }
        let output = std::process::Command::new("python3")
            .args(["-c", &format!(
                "from huggingface_hub import snapshot_download; \
                 d = snapshot_download('{repo}'); print(d)"
            )])
            .output();
        if let Ok(out) = output {
            if out.status.success() {
                let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
                let w   = dir.join("model-00001-of-00001.safetensors");
                let c   = dir.join("config.json");
                if w.exists() && c.exists() { return Ok((w, c)); }
            }
        }
        anyhow::bail!(
            "Could not locate weights for '{repo}'.\n\
             Options:\n  a) --features hf-download\n  b) --weights … --config …"
        )
    }
}

fn hf_cache_dir() -> PathBuf {
    std::env::var("HF_HOME").map(PathBuf::from).unwrap_or_else(|_| {
        std::env::var("HOME").map(|h| PathBuf::from(h)
            .join(".cache").join("huggingface").join("hub"))
            .unwrap_or_else(|_| PathBuf::from("/root/.cache/huggingface/hub"))
    })
}

fn chrono_now() -> String {
    // Simple ISO-8601 timestamp without chrono dep
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Days since epoch
    let d = secs / 86400;
    let y = 1970 + d / 365;  // rough — good enough for a label
    let rem = d % 365;
    let mo = rem / 30 + 1;
    let dy = rem % 30 + 1;
    let hh = (secs % 86400) / 3600;
    let mm = (secs % 3600)  / 60;
    let ss = secs % 60;
    format!("{y:04}-{mo:02}-{dy:02}T{hh:02}:{mm:02}:{ss:02}Z")
}
