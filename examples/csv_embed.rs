//! Embed EEG data from a CSV file using the ZUNA encoder.
//!
//! Demonstrates loading a CSV with configurable channel count and padding
//! strategy, then running the encoder to produce latent embeddings.
//!
//! # Usage
//!
//! ```sh
//! # Embed every channel found in the CSV
//! cargo run --example csv_embed --release -- --csv data/sample.csv
//!
//! # Take only the first 8 channels, zero-pad to the full 10-20 target
//! cargo run --example csv_embed --release -- \
//!     --csv data/sample.csv \
//!     --n-channels 8 \
//!     --target "Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8"
//!
//! # Clone the nearest real channel for any missing one
//! cargo run --example csv_embed --release -- \
//!     --csv data/sample.csv --strategy nearest
//!
//! # Save embeddings to a safetensors file
//! cargo run --example csv_embed --release -- \
//!     --csv data/sample.csv --output embeddings.safetensors
//!
//! # GPU (wgpu backend)
//! cargo run --example csv_embed --release \
//!     --no-default-features --features wgpu -- \
//!     --csv data/sample.csv --device gpu
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;

use burn::prelude::Backend;
use clap::{Parser, ValueEnum};
use zuna_rs::{
    config::DataConfig,

    load_from_csv, CsvLoadOptions, PaddingStrategy, ZunaEncoder,
};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum DeviceArg { Cpu, Gpu }

#[derive(Debug, Clone, ValueEnum)]
enum StrategyArg {
    /// Fill missing channels with all-zero signal.
    Zero,
    /// Clone data from the nearest loaded channel (by scalp XYZ distance).
    Nearest,
    /// Clone data from the Fp1 channel for all missing channels.
    CloneFp1,
    /// No synthesis — channels absent from the CSV are simply dropped.
    /// The encoder receives only the channels that are actually present.
    Native,
}

#[derive(Parser, Debug)]
#[command(
    name  = "csv_embed",
    about = "Embed EEG from CSV using the ZUNA encoder",
    after_help = "\
CHANNEL SELECTION
  --n-channels N
      Use only the first N columns from the CSV. The remaining CSV channels
      are discarded; if a --target list is given, any target names absent from
      those N channels will be synthesised according to --strategy.

  --target ch1,ch2,...
      Ordered list of channels expected by the encoder. Channels present in
      the CSV (within the first --n-channels) are copied; missing ones are
      padded according to --strategy. If omitted, all CSV channels are used.

PADDING STRATEGIES
  zero      All-zeros signal; position from montage database.
  nearest   Data from the closest loaded channel by XYZ distance.
  clone-fp1 Data from the Fp1 column (must be present in the CSV).
  native    No synthesis — drop missing channels; encoder sees only what is real.

WEIGHT RESOLUTION (priority)
  1. Both --weights and --config given explicitly.
  2. --features hf-download: hf-hub checks cache then downloads.
  3. Manual scan of ~/.cache/huggingface/hub/ for an existing snapshot.
  4. python3 huggingface_hub fallback.
"
)]
struct Args {
    /// Path to the EEG CSV file (timestamp column + channel columns).
    /// Defaults to data/sample.csv; auto-generated from data/sample1_raw.fif
    /// on first run if the file does not yet exist.
    #[arg(long, short = 'i',
          default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample.csv"))]
    csv: String,

    /// Use only the first N channels from the CSV (0 = all).
    #[arg(long, default_value_t = 0)]
    n_channels: usize,

    /// Ordered target channel list (comma-separated).
    /// Example: "Fp1,Fp2,F3,F4,C3,C4,P3,P4,O1,O2,F7,F8"
    #[arg(long, value_name = "CH1,CH2,...")]
    target: Option<String>,

    /// Strategy for filling channels absent from the CSV / not in --n-channels.
    #[arg(long, default_value = "zero")]
    strategy: StrategyArg,

    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: DeviceArg,

    /// Signal normalisation divisor (divide after z-score).
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Output CSV file for embeddings (one row per token, human-readable).
    #[arg(
        long,
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/csv_embeddings.csv")
    )]
    output: String,

    /// HuggingFace model repo ID.
    #[arg(long, default_value = "Zyphra/ZUNA", env = "ZUNA_REPO")]
    repo: String,

    /// Explicit safetensors weights path (skip HF resolution).
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<PathBuf>,

    /// Explicit config.json path (must pair with --weights).
    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<PathBuf>,
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

// ── Per-backend shims ─────────────────────────────────────────────────────────

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
// Generic runner
// ─────────────────────────────────────────────────────────────────────────────

fn run<B: Backend>(device: B::Device, args: Args) -> anyhow::Result<()>
where
    B::Device: Clone,
{
    let t_total = Instant::now();

    // ── 1. Resolve weights ────────────────────────────────────────────────────
    print!("[1/4] Resolving weights ... ");
    let (weights_path, config_path) =
        resolve_weights(&args.repo, args.weights, args.config)?;
    println!("OK\n  config  : {}\n  weights : {}",
        config_path.display(), weights_path.display());

    // ── 2. Load encoder ───────────────────────────────────────────────────────
    print!("[2/4] Loading encoder ... ");
    let t_load = Instant::now();
    let data_device = device.clone();
    let (enc, _) = ZunaEncoder::<B>::load(&config_path, &weights_path, device)?;
    println!("OK ({:.0} ms)\n  {}", t_load.elapsed().as_millis(), enc.describe());

    // ── 3. Load and preprocess CSV ────────────────────────────────────────────
    // Auto-generate data/sample.csv from the bundled FIF when the default path
    // is requested but the file doesn't exist yet.
    let default_csv = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample.csv");
    let default_fif = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif");
    if !std::path::Path::new(&args.csv).exists() {
        if args.csv == default_csv {
            println!("[auto] data/sample.csv not found — generating from data/sample1_raw.fif …");
            let (names, hz) = zuna_rs::fif_to_csv(
                std::path::Path::new(default_fif),
                std::path::Path::new(default_csv),
                None,
            )?;
            println!("[auto] wrote {} channels × {:.0} Hz → {default_csv}",
                names.len(), hz);
        } else {
            anyhow::bail!("CSV file not found: {}", args.csv);
        }
    }

    print!("[3/4] Loading CSV: {} ... ", args.csv);
    let t_pp = Instant::now();

    let padding = match args.strategy {
        StrategyArg::Zero     => PaddingStrategy::Zero,
        StrategyArg::Nearest  => PaddingStrategy::CloneNearest,
        StrategyArg::CloneFp1 => PaddingStrategy::CloneChannel("Fp1".to_string()),
        StrategyArg::Native   => PaddingStrategy::NoPadding,
    };

    let csv_path = Path::new(&args.csv);

    // Build target + optional whitelist for --n-channels restriction.
    let (final_target, whitelist) =
        build_target(csv_path, args.n_channels, args.target.as_deref())?;

    let opts = CsvLoadOptions {
        data_norm:         args.data_norm,
        target_channels:   final_target,
        padding,
        channel_whitelist: whitelist,
        ..CsvLoadOptions::default()
    };

    let (batches, info) =
        load_from_csv::<B>(csv_path, &opts, &DataConfig::default(), &data_device)?;
    println!("OK ({:.0} ms)", t_pp.elapsed().as_millis());
    println!("  recording : {:.2} s  →  {} epochs", info.duration_s, info.n_epochs);
    println!("  channels  : {} total  ({} from CSV, {} padded)",
        info.ch_names.len(),
        info.ch_names.len() - info.n_padded,
        info.n_padded);
    println!("  strategy  : {:?}", args.strategy);

    // Determine which channels are "real" (came from the CSV).
    // With a whitelist those are exactly the whitelisted names;
    // without a whitelist they are all CSV channel names.
    let real_name_set: std::collections::HashSet<String> = {
        let csv_header = std::fs::read_to_string(csv_path)
            .ok()
            .and_then(|c| {
                c.lines()
                    .find(|l| { let t = l.trim(); !t.is_empty() && !t.starts_with('#') })
                    .map(|h| h.to_string())
            })
            .unwrap_or_default();
        let all_csv: std::collections::HashSet<String> = csv_header
            .split(',').skip(1)
            .map(|s| s.trim().to_ascii_lowercase())
            .collect();

        if let Some(ref wl) = opts.channel_whitelist {
            wl.iter()
                .filter(|n| all_csv.contains(&n.to_ascii_lowercase()))
                .map(|n| n.to_ascii_lowercase())
                .collect()
        } else {
            all_csv
        }
    };

    let strat_name = match args.strategy {
        StrategyArg::Zero     => "zero",
        StrategyArg::Nearest  => "clone-nearest",
        StrategyArg::CloneFp1 => "clone-fp1",
        StrategyArg::Native   => "native",
    };

    println!("\n  {:12} {:8} {:8} {:8}  source",
        "name", "x[m]", "y[m]", "z[m]");
    println!("  {}", "─".repeat(58));
    for (name, pos) in info.ch_names.iter().zip(info.ch_pos_m.iter()) {
        let is_real = real_name_set.contains(&name.to_ascii_lowercase());
        let src     = if is_real { "real" } else { strat_name };
        println!("  {:12} {:+8.4} {:+8.4} {:+8.4}  {}",
            name, pos[0], pos[1], pos[2], src);
    }

    // ── 4. Encode ─────────────────────────────────────────────────────────────
    print!("\n[4/4] Encoding ... ");
    let t_enc = Instant::now();
    let epoch_embs = enc.encode_batches(batches)?;
    println!("OK ({:.0} ms)", t_enc.elapsed().as_millis());

    // ── Print embedding statistics ─────────────────────────────────────────────
    println!("\n  {:6}  {:8}  {:6}  {:8}  {:8}  {:8}  {:8}",
        "epoch", "tokens", "dims", "mean", "std", "min", "max");
    println!("  {}", "─".repeat(66));
    for (i, ep) in epoch_embs.iter().enumerate() {
        let v = &ep.embeddings;
        let n = v.len() as f64;
        let mean  = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let std   = (v.iter().map(|&x| { let d = x as f64 - mean; d*d }).sum::<f64>() / n).sqrt();
        let min   = v.iter().cloned().fold( f32::INFINITY, f32::min);
        let max   = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  {:6}  {:8}  {:6}  {:+8.4}  {:8.4}  {:+8.4}  {:+8.4}",
            i, ep.n_tokens(), ep.output_dim(), mean, std, min, max);
    }

    // ── Save embeddings as human-readable CSV ─────────────────────────────────
    let out_path = std::path::Path::new(&args.output);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_embeddings_csv(
        out_path,
        &epoch_embs,
        &info.ch_names,
        &info.ch_pos_m,
        &args.csv,
    )?;
    println!("\nEmbeddings saved → {}", args.output);

    println!("\nTotal: {:.0} ms", t_total.elapsed().as_millis());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Write embeddings to a human-readable CSV file.
///
/// ## Format
/// ```text
/// # zuna-rs csv_embed  source: data/sample.csv
/// # n_epochs: 3  n_channels: 12  tc: 40  output_dim: 32
/// epoch,channel,t_coarse,x_m,y_m,z_m,x_bin,y_bin,z_bin,emb_0,...,emb_31
/// 0,Fp1,0,-0.025700,0.073400,-0.006100,12,18,3,0.123456,...
/// ```
///
/// Rows are ordered: epoch → channel → t_coarse (same order as the encoder
/// output).  Each row is one token; token `i` in epoch `e` maps to:
/// - `channel = ch_names[i / tc]`
/// - `t_coarse = i % tc`
fn save_embeddings_csv(
    path:      &Path,
    epochs:    &[zuna_rs::EpochEmbedding],
    ch_names:  &[String],
    ch_pos_m:  &[[f32; 3]],
    source:    &str,
) -> anyhow::Result<()> {
    use std::io::{BufWriter, Write as IoWrite};

    let f   = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);

    let n_epochs   = epochs.len();
    let n_channels = epochs.first().map(|e| e.n_channels).unwrap_or(0);
    let tc         = epochs.first().map(|e| e.tc).unwrap_or(0);
    let out_dim    = epochs.first().map(|e| e.output_dim()).unwrap_or(0);

    // ── Comment header ────────────────────────────────────────────────────────
    writeln!(w, "# zuna-rs csv_embed  source: {source}")?;
    writeln!(w, "# n_epochs: {n_epochs}  n_channels: {n_channels}  \
                 tc: {tc}  output_dim: {out_dim}")?;

    // ── Column header ─────────────────────────────────────────────────────────
    write!(w, "epoch,channel,t_coarse,x_m,y_m,z_m,x_bin,y_bin,z_bin")?;
    for d in 0..out_dim { write!(w, ",emb_{d}")?; }
    writeln!(w)?;

    // ── Data rows ─────────────────────────────────────────────────────────────
    for (ep_idx, ep) in epochs.iter().enumerate() {
        let n_tok = ep.n_tokens();
        let dim   = ep.output_dim();

        for ti in 0..n_tok {
            let ci      = ti / ep.tc;           // channel index
            let t_c     = ti % ep.tc;           // coarse time step
            let ch_name = ch_names.get(ci).map(|s| s.as_str()).unwrap_or("?");

            // XYZ position in metres
            let (xm, ym, zm) = ch_pos_m.get(ci)
                .map(|&p| (p[0], p[1], p[2]))
                .unwrap_or((0.0, 0.0, 0.0));

            // Discretised position bins + t_coarse from tok_idx [x,y,z,tc]
            let base4    = ti * 4;
            let x_bin = ep.tok_idx.get(base4)  .copied().unwrap_or(0);
            let y_bin = ep.tok_idx.get(base4+1).copied().unwrap_or(0);
            let z_bin = ep.tok_idx.get(base4+2).copied().unwrap_or(0);

            // Embedding values
            let emb_base = ti * dim;

            write!(w, "{ep_idx},{ch_name},{t_c},{xm:.6},{ym:.6},{zm:.6},\
                       {x_bin},{y_bin},{z_bin}")?;
            for d in 0..dim {
                let v = ep.embeddings.get(emb_base + d).copied().unwrap_or(0.0);
                write!(w, ",{v:.6}")?;
            }
            writeln!(w)?;
        }
    }

    Ok(())
}

/// Determine the final target channel list and optional whitelist.
///
/// Returns `(target_channels, channel_whitelist)`.
///
/// | --n-channels | --target | target                     | whitelist          |
/// |---|---|---|---|
/// | 0            | None     | None (use all CSV channels)| None               |
/// | 0            | Some(T)  | Some(T)                    | None               |
/// | N > 0        | None     | Some(first_N from CSV)     | None               |
/// | N > 0        | Some(T)  | Some(T)                    | Some(first_N)      |
///
/// When a whitelist is set, the loader treats only those N channels as
/// "present" in the CSV — all others (even if they exist in the file) are
/// synthesised via the padding strategy.
fn build_target(
    csv_path:        &Path,
    n_channels:      usize,
    explicit_target: Option<&str>,
) -> anyhow::Result<(Option<Vec<String>>, Option<Vec<String>>)> {
    let target: Option<Vec<String>> = explicit_target.map(|s| {
        s.split(',')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect()
    });

    if n_channels == 0 {
        return Ok((target, None));
    }

    // Read CSV header to get channel names
    let content = std::fs::read_to_string(csv_path)?;
    let header  = content
        .lines()
        .find(|l| { let t = l.trim(); !t.is_empty() && !t.starts_with('#') })
        .ok_or_else(|| anyhow::anyhow!("CSV file is empty or has no header"))?;

    let all_csv_channels: Vec<String> = header
        .split(',')
        .skip(1)  // skip timestamp column
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    anyhow::ensure!(
        n_channels <= all_csv_channels.len(),
        "--n-channels {} exceeds the CSV channel count ({})",
        n_channels, all_csv_channels.len()
    );

    let first_n: Vec<String> = all_csv_channels[..n_channels].to_vec();

    match target {
        // --n-channels N, --target T:
        //   target = T (use the full target layout)
        //   whitelist = first N CSV channels (only those count as "real")
        Some(t) => Ok((Some(t), Some(first_n))),

        // --n-channels N, no --target:
        //   target = first N channels (no padding — all are present)
        //   whitelist = not needed
        None => Ok((Some(first_n), None)),
    }
}

/// Resolve model weights (same logic as embedding_api.rs).
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

    // 1. hf-hub (if compiled in)
    #[cfg(feature = "hf-download")]
    {
        use hf_hub::api::sync::ApiBuilder;
        let model   = ApiBuilder::new().with_progress(true).build()?.model(repo.to_string());
        let weights = model.get("model-00001-of-00001.safetensors")?;
        let config  = model.get("config.json")?;
        return Ok((weights, config));
    }

    // 2. Scan HuggingFace local cache
    #[allow(unreachable_code)]
    {
        let slug = repo.replace('/', "--");
        let hf_dir = dirs_or_default().join(format!("models--{slug}/snapshots"));
        if hf_dir.is_dir() {
            for entry in std::fs::read_dir(&hf_dir)?.flatten() {
                let snap = entry.path();
                let w = snap.join("model-00001-of-00001.safetensors");
                let c = snap.join("config.json");
                if w.exists() && c.exists() {
                    println!("  (found cached snapshot: {})", snap.display());
                    return Ok((w, c));
                }
            }
        }

        // 3. python3 fallback
        let output = std::process::Command::new("python3")
            .args(["-c",
                &format!(
                    "from huggingface_hub import snapshot_download; \
                     import json, sys; \
                     d = snapshot_download('{repo}'); \
                     print(d)"
                ),
            ])
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
            "Could not locate ZUNA weights for '{repo}'.\n\
             Options:\n  \
             a) cargo run --features hf-download\n  \
             b) --weights model.safetensors --config config.json\n  \
             c) pip install huggingface_hub && huggingface-cli download {repo}"
        )
    }
}

fn dirs_or_default() -> PathBuf {
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next_home()
                .join(".cache")
                .join("huggingface")
                .join("hub")
        })
}

fn dirs_next_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
}
