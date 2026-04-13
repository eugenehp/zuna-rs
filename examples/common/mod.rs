//! Shared utilities for ZUNA examples.
#![allow(dead_code)]
//!
//! Included by each example with:
//!   `#[path = "common/mod.rs"] mod common;`

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use plotters::prelude::*;

// ── Constants ─────────────────────────────────────────────────────────────────

pub const DEFAULT_REPO: &str = "Zyphra/ZUNA";
pub const WEIGHTS_FILE: &str = "model-00001-of-00001.safetensors";
pub const CONFIG_FILE:  &str = "config.json";

/// Tab10 palette (matches matplotlib defaults).
const PALETTE: &[RGBColor] = &[
    RGBColor( 31, 119, 180),   // blue
    RGBColor(255, 127,  14),   // orange
    RGBColor( 44, 160,  44),   // green
    RGBColor(214,  39,  40),   // red
    RGBColor(148, 103, 189),   // purple
    RGBColor(140,  86,  75),   // brown
    RGBColor(227, 119, 194),   // pink
    RGBColor(127, 127, 127),   // grey
    RGBColor(188, 189,  34),   // yellow-green
    RGBColor( 23, 190, 207),   // cyan
];

// ── HuggingFace weight resolution ─────────────────────────────────────────────

/// Resolve paths to `weights.safetensors` and `config.json`.
///
/// Priority:
///   1. Both `--weights` and `--config` provided explicitly → use them.
///   2. `hf-download` feature compiled in → download/cache via `hf-hub`.
///   3. Scan `~/.cache/huggingface/hub/` for an existing snapshot.
pub fn resolve_weights(
    repo_id:          &str,
    weights_override: Option<&str>,
    config_override:  Option<&str>,
    cache_dir:        Option<&Path>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    match (weights_override, config_override) {
        (Some(w), Some(c)) => return Ok((w.into(), c.into())),
        (Some(_), None) | (None, Some(_)) => anyhow::bail!(
            "supply both --weights and --config together, or neither (auto-resolve from HF)"
        ),
        (None, None) => {}
    }

    // ── Try hf-hub (auto-download, requires `hf-download` feature) ────────────
    #[cfg(feature = "hf-download")]
    {
        match hf_hub_resolve(repo_id, cache_dir) {
            Ok(p)  => return Ok(p),
            Err(e) => eprintln!("⚠  hf-hub: {e}  — falling back to cache scan"),
        }
    }

    // ── Scan local HF disk cache ──────────────────────────────────────────────
    match scan_hf_cache(repo_id, cache_dir) {
        Ok(paths) => return Ok(paths),
        Err(_)    => {}
    }

    // ── Last resort: download via Python huggingface_hub ─────────────────────
    println!("  Model not in local cache — downloading via Python huggingface_hub …");
    println!("  (This may take several minutes for a 1.7 GB model)");
    download_via_python(repo_id)?;

    // Now the cache scan should succeed
    scan_hf_cache(repo_id, cache_dir)
}

#[cfg(feature = "hf-download")]
fn hf_hub_resolve(repo_id: &str, cache_dir: Option<&Path>) -> anyhow::Result<(PathBuf, PathBuf)> {
    use hf_hub::api::sync::ApiBuilder;
    let mut b = ApiBuilder::new().with_progress(true);
    if let Some(d) = cache_dir { b = b.with_cache_dir(d.to_path_buf()); }
    let repo = b.build()?.model(repo_id.to_string());
    Ok((repo.get(WEIGHTS_FILE)?, repo.get(CONFIG_FILE)?))
}

fn scan_hf_cache(repo_id: &str, cache_dir: Option<&Path>) -> anyhow::Result<(PathBuf, PathBuf)> {
    let base = cache_dir.map(PathBuf::from).unwrap_or_else(default_hf_cache);
    // HF uses "--" as path separator: "Zyphra/ZUNA" → "models--Zyphra--ZUNA"
    let snapshots = base
        .join(format!("models--{}", repo_id.replace('/', "--")))
        .join("snapshots");

    anyhow::ensure!(
        snapshots.exists(),
        "HuggingFace cache not found at {snapshots:?}.\n\
         Download the model first:\n  \
         python3 -c \"from huggingface_hub import snapshot_download; \
         snapshot_download('{repo_id}')\"\n\
         (use 'python' instead of 'python3' if that is your interpreter name)",
    );

    // Pick the most-recently-modified snapshot hash directory.
    let mut dirs: Vec<_> = std::fs::read_dir(&snapshots)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    dirs.sort_by_key(|e| {
        e.metadata().and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    let snap = dirs.last()
        .ok_or_else(|| anyhow::anyhow!("no snapshot dirs under {snapshots:?}"))?
        .path();

    let w = snap.join(WEIGHTS_FILE);
    let c = snap.join(CONFIG_FILE);
    anyhow::ensure!(w.exists(), "weights not in snapshot: {w:?}");
    anyhow::ensure!(c.exists(), "config  not in snapshot: {c:?}");
    Ok((w, c))
}

/// Find a working Python interpreter.
///
/// Check order:
///   1. `$ZUNA_PYTHON` env var (set by `benchmark.sh` to match the shell's python)
///   2. `python3`
///   3. `python`
fn find_python() -> Option<String> {
    if let Ok(py) = std::env::var("ZUNA_PYTHON") {
        let py = py.trim().to_string();
        if !py.is_empty() {
            return Some(py);
        }
    }
    for candidate in ["python3", "python"] {
        let ok = std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok { return Some(candidate.to_string()); }
    }
    None
}

/// Download a HuggingFace model snapshot using Python's `huggingface_hub`.
///
/// Tries `$ZUNA_PYTHON`, then `python3`, then `python`.
/// Uses `.output()` (not `.status()`) so that conda/pip noise printed to stdout
/// by the interpreter on macOS does not pollute parseable output.
fn download_via_python(repo_id: &str) -> anyhow::Result<()> {
    let python = find_python().ok_or_else(|| anyhow::anyhow!(
        "no Python interpreter found (tried: $ZUNA_PYTHON, python3, python)\n\
         Install huggingface_hub: pip install huggingface_hub"
    ))?;

    let code = format!(
        "from huggingface_hub import snapshot_download; \
         snap = snapshot_download('{repo_id}'); \
         import sys; sys.stderr.write(f'Downloaded to: {{snap}}\\n')"
    );

    // Capture stdout so conda activation messages (e.g. "Requirement already
    // satisfied: numpy") don't leak into terminal output.
    // Stream stderr so HuggingFace progress bars remain visible.
    let out = std::process::Command::new(&python)
        .args(["-c", &code])
        .env("PYTHONWARNINGS", "ignore")
        .output()
        .with_context(|| format!("{python} failed to start"))?;

    std::io::stderr().write_all(&out.stderr).ok();

    if !out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!(
            "snapshot_download failed (exit {:?}):\nstdout: {stdout}\nstderr: {stderr}",
            out.status.code()
        );
    }
    Ok(())
}

fn default_hf_cache() -> PathBuf {
    // Respect $HF_HOME first, then $XDG_CACHE_HOME, then ~/.cache
    if let Ok(v) = std::env::var("HF_HOME") { return PathBuf::from(v).join("hub"); }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

pub fn ensure_figures_dir(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating figures dir {}", dir.display()))
}

// ── Step-by-step timer ────────────────────────────────────────────────────────

pub struct StepTimer {
    run_start:  Instant,
    step_start: Instant,
    step:       usize,
    total:      usize,
    pub verbose: bool,
}

impl StepTimer {
    pub fn new(total: usize, verbose: bool) -> Self {
        let now = Instant::now();
        Self { run_start: now, step_start: now, step: 0, total, verbose }
    }

    /// Mark the beginning of a step and print its header.
    pub fn begin(&mut self, desc: &str) {
        self.step += 1;
        self.step_start = Instant::now();
        if self.verbose {
            println!("\n[{:>6.2}s] ▶ [{}/{}] {desc}",
                self.run_start.elapsed().as_secs_f64(), self.step, self.total);
        } else {
            print!("[{}/{}] {desc} … ", self.step, self.total);
            let _ = std::io::stdout().flush();
        }
    }

    /// Mark this step as done.  Returns elapsed ms for this step.
    pub fn done(&self, detail: &str) -> f64 {
        let ms = self.step_start.elapsed().as_secs_f64() * 1000.0;
        if self.verbose {
            println!("[{:>6.2}s] ✓  {ms:.0} ms  {detail}",
                self.run_start.elapsed().as_secs_f64());
        } else {
            println!("{ms:.0} ms  {detail}");
        }
        ms
    }

    /// Print a sub-step note (only shown in verbose mode).
    pub fn sub(&self, msg: &str) {
        if self.verbose {
            println!("[{:>6.2}s]   {msg}", self.run_start.elapsed().as_secs_f64());
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.run_start.elapsed().as_secs_f64() * 1000.0
    }
}

// ── Chart helpers ─────────────────────────────────────────────────────────────
//
// All functions create a PNG at `path` and print "chart → <path>" on success.

// ── 1. Timing breakdown (horizontal bars) ─────────────────────────────────────

/// Horizontal bar chart showing how long each step took.
/// `items`: `&[("Step label", milliseconds)]`
pub fn save_timing_chart(
    path:  &Path,
    title: &str,
    items: &[(&str, f64)],
) -> anyhow::Result<()> {
    const W:      u32 = 900;
    const ROW_H:  i32 = 40;
    const GAP:    i32 = 14;
    const TOP:    i32 = 64;   // space for title
    const LEFT:   i32 = 170;  // label column
    const RPAD:   i32 = 130;  // value text column
    let bar_area  = W as i32 - LEFT - RPAD;
    let h         = (TOP + items.len() as i32 * (ROW_H + GAP) + 30) as u32;

    let root = BitMapBackend::new(path, (W, h)).into_drawing_area();
    root.fill(&WHITE)?;

    root.draw(&Text::new(
        title,
        (W as i32 / 2 - title.len() as i32 * 6, 22),
        ("sans-serif", 20u32).into_font(),
    ))?;

    let max_ms = items.iter().map(|(_, v)| *v).fold(1.0f64, f64::max);

    for (i, (label, ms)) in items.iter().enumerate() {
        let y    = TOP + i as i32 * (ROW_H + GAP);
        let fill = ((ms / max_ms) * bar_area as f64).max(2.0) as i32;
        let col  = PALETTE[i % PALETTE.len()];

        // Label (approximate right-align by shifting left by string width)
        root.draw(&Text::new(
            *label,
            (LEFT - label.len() as i32 * 8 - 6, y + ROW_H / 2 - 8),
            ("sans-serif", 14u32).into_font(),
        ))?;

        // Filled bar
        root.draw(&Rectangle::new(
            [(LEFT, y + 3), (LEFT + fill, y + ROW_H - 3)],
            ShapeStyle::from(&col).filled(),
        ))?;

        // Value label to the right of the bar
        root.draw(&Text::new(
            format!("{ms:.0} ms"),
            (LEFT + fill + 6, y + ROW_H / 2 - 8),
            ("sans-serif", 13u32).into_font(),
        ))?;
    }

    root.present()?;
    println!("  chart → {}", path.display());
    Ok(())
}

// ── 2. Multi-channel EEG waveform ─────────────────────────────────────────────

/// Line chart of reconstructed EEG signal.
/// `signal`: `[n_channels][n_timesteps]`.  At most 8 channels are shown.
pub fn save_waveform_chart(
    path:     &Path,
    title:    &str,
    signal:   &[Vec<f32>],
    ch_names: &[String],
    sfreq:    f32,
) -> anyhow::Result<()> {
    let n_ch = signal.len().min(8);
    let n_t  = signal[0].len();
    let dur  = n_t as f64 / sfreq as f64;

    let (y_lo, y_hi) = signal[..n_ch].iter()
        .flat_map(|ch| ch.iter().copied())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let pad = (y_hi - y_lo).max(0.1) * 0.06;

    let root = BitMapBackend::new(path, (1100, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18u32).into_font())
        .margin(20i32)
        .x_label_area_size(46i32)
        .y_label_area_size(64i32)
        .build_cartesian_2d(0.0f64..dur, (y_lo - pad) as f64..(y_hi + pad) as f64)?;

    chart.configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Amplitude")
        .x_labels(10).y_labels(6)
        .draw()?;

    for (ch, samples) in signal[..n_ch].iter().enumerate() {
        let col   = PALETTE[ch % PALETTE.len()];
        let label = ch_names.get(ch).map(|s| s.as_str()).unwrap_or("?");
        chart.draw_series(LineSeries::new(
            samples.iter().enumerate()
                .map(|(t, &v)| (t as f64 / sfreq as f64, v as f64)),
            col.stroke_width(1),
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], col.stroke_width(2)));
    }

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 11u32).into_font())
        .draw()?;

    root.present()?;
    println!("  chart → {}", path.display());
    Ok(())
}

// ── 3. Per-epoch reconstruction statistics ────────────────────────────────────

/// Grouped bar chart: mean / ±std / min-max per epoch.
pub fn save_epoch_stats_chart(
    path:  &Path,
    title: &str,
    means: &[f64],
    stds:  &[f64],
    mins:  &[f64],
    maxs:  &[f64],
) -> anyhow::Result<()> {
    let n     = means.len();
    let y_abs = maxs.iter().copied().fold(0.0f64, f64::max)
        .max(mins.iter().copied().fold(0.0f64, |a, v| a.max(v.abs())));
    let y_max =  y_abs * 1.1;
    let y_min = -y_abs * 1.1;

    let root = BitMapBackend::new(path, (800, 450)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18u32).into_font())
        .margin(20i32)
        .x_label_area_size(46i32)
        .y_label_area_size(64i32)
        .build_cartesian_2d(0.0f64..(n as f64 + 0.2), y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Value")
        .x_labels(n.max(1)).y_labels(7)
        .draw()?;

    // Min-max range (narrow bar)
    chart.draw_series(mins.iter().zip(maxs.iter()).enumerate().map(|(i, (&mn, &mx))| {
        let xc = i as f64 + 0.5;
        Rectangle::new([(xc - 0.08, mn), (xc + 0.08, mx)],
            RGBColor(180, 180, 220).mix(0.7).filled())
    }))?;

    // ±std bar
    chart.draw_series(means.iter().zip(stds.iter()).enumerate().map(|(i, (&m, &s))| {
        let xc = i as f64 + 0.5;
        Rectangle::new([(xc - 0.22, m - s), (xc + 0.22, m + s)],
            PALETTE[1].mix(0.55).filled())
    }))?
    .label("±std")
    .legend(|(x, y)| Rectangle::new([(x, y - 4), (x + 12, y + 4)], PALETTE[1].mix(0.55).filled()));

    // Mean line marker
    let tick = (y_max - y_min) * 0.005;
    chart.draw_series(means.iter().enumerate().map(|(i, &m)| {
        let xc = i as f64 + 0.5;
        Rectangle::new([(xc - 0.30, m - tick), (xc + 0.30, m + tick)], PALETTE[0].filled())
    }))?
    .label("mean")
    .legend(|(x, y)| Rectangle::new([(x, y - 4), (x + 12, y + 4)], PALETTE[0].filled()));

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 11u32).into_font())
        .draw()?;

    root.present()?;
    println!("  chart → {}", path.display());
    Ok(())
}

// ── 4. Embedding value distribution histogram ─────────────────────────────────

/// Histogram of all embedding values with an N(0,1) reference curve.
pub fn save_distribution_chart(
    path:   &Path,
    title:  &str,
    values: &[f32],
    n_bins: usize,
) -> anyhow::Result<()> {
    let lo = (values.iter().copied().fold(f32::INFINITY, f32::min) as f64).min(-4.5);
    let hi = (values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64).max(4.5);
    let bw = (hi - lo) / n_bins as f64;

    let mut counts = vec![0u32; n_bins];
    for &v in values {
        let b = (((v as f64 - lo) / bw) as usize).min(n_bins - 1);
        counts[b] += 1;
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;

    let root = BitMapBackend::new(path, (900, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18u32).into_font())
        .margin(20i32)
        .x_label_area_size(46i32)
        .y_label_area_size(64i32)
        .build_cartesian_2d(lo..hi, 0.0f64..(max_c * 1.12))?;

    chart.configure_mesh()
        .x_desc("Embedding value")
        .y_desc("Count")
        .x_labels(9).y_labels(6)
        .draw()?;

    // Histogram bins
    chart.draw_series(
        counts.iter().enumerate().map(|(i, &c)| {
            let x0 = lo + i as f64 * bw;
            Rectangle::new([(x0, 0.0), (x0 + bw, c as f64)], PALETTE[0].mix(0.72).filled())
        })
    )?;

    // N(0,1) reference curve scaled to histogram area
    let scale = values.len() as f64 * bw;
    chart.draw_series(LineSeries::new(
        (0..=300).map(|k| {
            let x = lo + (hi - lo) * k as f64 / 300.0;
            let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            (x, scale * pdf)
        }),
        PALETTE[3].stroke_width(2),
    ))?
    .label("N(0,1) ideal")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], PALETTE[3].stroke_width(2)));

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 11u32).into_font())
        .draw()?;

    root.present()?;
    println!("  chart → {}", path.display());
    Ok(())
}

// ── 5. Per-latent-dimension mean ± std ────────────────────────────────────────

/// Bar chart of per-dimension mean (across all tokens) with ±std bands.
/// Ideal for verifying MMD regularlisation → means ≈ 0, stds ≈ 1.
pub fn save_dim_stats_chart(
    path:      &Path,
    title:     &str,
    dim_means: &[f64],
    dim_stds:  &[f64],
) -> anyhow::Result<()> {
    let n     = dim_means.len();
    let y_abs = dim_means.iter().zip(dim_stds.iter())
        .map(|(m, s)| m.abs() + s)
        .fold(1.0f64, f64::max) * 1.15;

    let root = BitMapBackend::new(path, (1100, 450)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 18u32).into_font())
        .margin(20i32)
        .x_label_area_size(46i32)
        .y_label_area_size(64i32)
        .build_cartesian_2d(0.0f64..(n as f64), -y_abs..y_abs)?;

    chart.configure_mesh()
        .x_desc("Latent dimension")
        .y_desc("Mean across tokens")
        .x_labels(n.min(16)).y_labels(7)
        .draw()?;

    // ±std shaded band
    chart.draw_series(
        dim_means.iter().zip(dim_stds.iter()).enumerate().map(|(i, (&m, &s))| {
            Rectangle::new(
                [(i as f64 + 0.05, m - s), (i as f64 + 0.95, m + s)],
                PALETTE[0].mix(0.22).filled(),
            )
        })
    )?
    .label("±std (tokens)")
    .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], PALETTE[0].mix(0.3).filled()));

    // Mean bars (blue = positive, red = negative)
    chart.draw_series(
        dim_means.iter().enumerate().map(|(i, &m)| {
            let col = if m >= 0.0 { PALETTE[0] } else { PALETTE[3] };
            let (y0, y1) = if m >= 0.0 { (0.0, m) } else { (m, 0.0) };
            Rectangle::new([(i as f64 + 0.2, y0), (i as f64 + 0.8, y1)], col.mix(0.85).filled())
        })
    )?
    .label("dim mean")
    .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 12, y + 5)], PALETTE[0].filled()));

    // Zero reference line
    chart.draw_series(std::iter::once(
        PathElement::new(vec![(0.0, 0.0), (n as f64, 0.0)], BLACK.stroke_width(1)),
    ))?;

    chart.configure_series_labels()
        .background_style(WHITE.mix(0.85).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 11u32).into_font())
        .draw()?;

    root.present()?;
    println!("  chart → {}", path.display());
    Ok(())
}
