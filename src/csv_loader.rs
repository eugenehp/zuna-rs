//! CSV and raw-tensor loading for ZUNA inference.
//!
//! Three entry points, all producing the same `Vec<InputBatch<B>>` that
//! [`ZunaEncoder`](crate::encoder::ZunaEncoder) consumes:
//!
//! | Function | Input |
//! |---|---|
//! | [`load_from_csv`] | CSV file: timestamp column + channel columns |
//! | [`load_from_raw_tensor`] | `ndarray::Array2<f32>` + explicit `[f32;3]` positions |
//! | [`load_from_named_tensor`] | `ndarray::Array2<f32>` + channel names (auto-lookup) |
//!
//! ## CSV format
//!
//! ```text
//! timestamp,Fp1,Fp2,F3,F4,C3,C4
//! 0.000000000e0,2.0721e-05,8.38e-07,...
//! 3.906250000e-3,...
//! ```
//!
//! - First column must be timestamps in **seconds** (column name is ignored;
//!   any leading column whose name contains "time" or is index 0 is treated as
//!   the timestamp).
//! - Remaining columns are EEG channel values in **volts**.
//! - Lines starting with `#` are ignored.
//! - Scientific notation (`1.23e-5`) and plain decimals both accepted.
//!
//! ## Padding
//!
//! When `target_channels` is set in [`CsvLoadOptions`], channels present in
//! the target list but absent from the CSV are synthesised:
//!
//! | [`PaddingStrategy`] | Data | Position |
//! |---|---|---|
//! | `Zero` | all-zero row | overrides → database → centroid |
//! | `CloneChannel(src)` | copy of the named channel's row | overrides → database → src's pos |
//! | `CloneNearest` | copy of nearest loaded channel by xyz | overrides → database → centroid |
//! | `InterpWeighted { k }` | inverse-distance–weighted mean of k nearest real channels | same as CloneNearest |
//! | `Mirror` | copy of nearest real channel on the opposite hemisphere (X flipped) | database → centroid |
//! | `MeanRef` | per-sample mean of all real channels (common average reference) | database → centroid |
//! | `NoPadding` | missing channels are **dropped** — output has fewer channels than the target list | n/a |

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context};
use burn::prelude::*;
use ndarray::Array2;

use crate::channel_positions::{channel_xyz, nearest_channel, normalise};
use crate::config::DataConfig;
use crate::data::{build_tok_idx, chop_and_reshape, discretize_chan_pos, InputBatch};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// How to synthesise EEG channels that are missing from the CSV.
#[derive(Debug, Clone)]
pub enum PaddingStrategy {
    /// Fill the missing channel with zeros.
    /// Its scalp position is taken from `position_overrides`, then the
    /// channel-position database, then the centroid of existing channels.
    Zero,

    /// Clone the data from a specific named channel.
    /// Position of the new channel: `position_overrides[missing]` →
    /// database lookup of the *missing* channel name → centroid.
    CloneChannel(String),

    /// Clone the data from whichever loaded channel is nearest (by Euclidean
    /// distance) to the missing channel's known position.
    /// Position of the new channel: `position_overrides[missing]` →
    /// database lookup of the *missing* channel name → centroid.
    CloneNearest,

    /// Synthesise by inverse-distance–weighted averaging of the `k` nearest
    /// real channels.  Uses all real channels when `k` ≥ number of real
    /// channels.  This is a simple form of scalp-surface interpolation.
    /// Position: same as [`CloneNearest`](Self::CloneNearest).
    InterpWeighted { k: usize },

    /// Copy the signal of the nearest real channel on the **opposite**
    /// hemisphere (the target channel's X coordinate is negated to find
    /// the "mirror" point, then the closest real channel to that point is
    /// used).  Useful for symmetric montages where the contralateral
    /// homologue is the best available substitute.
    /// Position: database → centroid.
    Mirror,

    /// Fill with the per-sample mean across **all** real channels.
    /// This is equivalent to injecting the common-average-reference (CAR)
    /// signal, which is the least-informative but spectrally neutral choice.
    /// Position: database → centroid.
    MeanRef,

    /// **No padding** — channels that are absent from the CSV are silently
    /// dropped from the output instead of being synthesised.
    ///
    /// The returned data will have fewer channels than `target_channels` when
    /// any targets are missing.  The encoder handles variable-length inputs
    /// natively, so the resulting [`InputBatch`](crate::data::InputBatch) is
    /// fully valid.
    NoPadding,
}

impl Default for PaddingStrategy {
    fn default() -> Self { Self::Zero }
}

/// Options for [`load_from_csv`].
#[derive(Debug, Clone)]
pub struct CsvLoadOptions {
    /// Sampling rate of the CSV data in Hz.  Default: `256.0`.
    pub sample_rate: f32,

    /// Signal normalisation divisor applied after z-scoring.  Default: `10.0`.
    pub data_norm: f32,

    /// If set, the output channels are reordered / padded to match this list.
    /// Channels in the CSV but *not* in this list are discarded.
    /// Channels in the list but *not* in the CSV are synthesised with [`padding`](Self::padding).
    pub target_channels: Option<Vec<String>>,

    /// Strategy for synthesising missing channels.  Default: [`PaddingStrategy::Zero`].
    pub padding: PaddingStrategy,

    /// Per-channel XYZ position overrides (metres).
    ///
    /// Keys are matched case-insensitively.  Use this to supply
    /// *fuzzy coordinates* for channels not in the standard montage database,
    /// or to override database positions for `CloneNearest` distance queries.
    pub position_overrides: HashMap<String, [f32; 3]>,

    /// If set, only CSV columns whose normalised name appears in this list are
    /// treated as **present**.  Other CSV columns are silently ignored — they
    /// will be synthesised as missing channels if they appear in
    /// `target_channels`.
    ///
    /// Use this to simulate recordings with fewer channels without modifying
    /// the CSV file (e.g. `--n-channels 6` in the `csv_embed` example).
    pub channel_whitelist: Option<Vec<String>>,
}

impl Default for CsvLoadOptions {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            data_norm:   10.0,
            target_channels:    None,
            padding:            PaddingStrategy::Zero,
            position_overrides: HashMap::new(),
            channel_whitelist:  None,
        }
    }
}

/// Metadata returned alongside the batches by [`load_from_csv`].
#[derive(Debug)]
pub struct CsvInfo {
    /// Final channel names after reordering and padding.
    pub ch_names: Vec<String>,
    /// Scalp positions in metres `[C, 3]` after reordering and padding.
    pub ch_pos_m: Vec<[f32; 3]>,
    /// Sample rate used (from [`CsvLoadOptions::sample_rate`]).
    pub sample_rate: f32,
    /// Number of raw time-samples read from the CSV.
    pub n_samples_raw: usize,
    /// Recording duration in seconds.
    pub duration_s: f32,
    /// Number of 5-second epochs produced.
    pub n_epochs: usize,
    /// Number of channels added by padding.
    pub n_padded: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point 1 — CSV file
// ─────────────────────────────────────────────────────────────────────────────

/// Load EEG data from a CSV file and run the full ZUNA preprocessing pipeline.
///
/// The pipeline is identical to [`load_from_fif`](crate::data::load_from_fif):
/// resample (if needed) → 0.5 Hz highpass FIR → average reference →
/// global z-score → epoch (5 s) → baseline correction → ÷ data_norm.
pub fn load_from_csv<B: Backend>(
    path:     &Path,
    opts:     &CsvLoadOptions,
    data_cfg: &DataConfig,
    device:   &B::Device,
) -> anyhow::Result<(Vec<InputBatch<B>>, CsvInfo)> {
    // ── Parse CSV ─────────────────────────────────────────────────────────────
    let (csv_names, raw_data) = parse_csv(path)
        .with_context(|| format!("parsing CSV {}", path.display()))?;
    let (_n_ch_raw, n_t) = raw_data.dim();

    // ── Look up positions for loaded channels ─────────────────────────────────
    let raw_positions = resolve_positions(&csv_names, &opts.position_overrides);

    // ── Apply target-channel reordering / padding ─────────────────────────────
    let (padded_data, padded_names, padded_positions, n_padded) =
        if let Some(ref targets) = opts.target_channels {
            apply_padding(
                &raw_data,
                &csv_names,
                &raw_positions,
                targets,
                &opts.padding,
                &opts.position_overrides,
                opts.channel_whitelist.as_deref(),
            )?
        } else if let Some(ref wl) = opts.channel_whitelist {
            // No explicit target — whitelist acts as the target list itself
            apply_padding(
                &raw_data,
                &csv_names,
                &raw_positions,
                wl,
                &opts.padding,
                &opts.position_overrides,
                Some(wl),
            )?
        } else {
            (raw_data, csv_names.clone(), raw_positions, 0)
        };

    let n_ch_final = padded_data.nrows();
    let duration_s = n_t as f32 / opts.sample_rate;

    // ── Minimum epoch size guard ──────────────────────────────────────────────
    let min_dur = 5.0_f32;
    if duration_s < min_dur {
        bail!(
            "CSV recording is {duration_s:.2} s, shorter than the minimum \
             epoch duration of {min_dur} s"
        );
    }

    // ── Run exg preprocessing pipeline ───────────────────────────────────────
    let pos_arr = positions_to_array(&padded_positions, n_ch_final);
    let batches = run_pipeline(
        padded_data, pos_arr, opts.sample_rate, opts.data_norm, data_cfg, device,
    )?;
    let n_epochs = batches.len();

    let info = CsvInfo {
        ch_names:      padded_names,
        ch_pos_m:      padded_positions,
        sample_rate:   opts.sample_rate,
        n_samples_raw: n_t,
        duration_s,
        n_epochs,
        n_padded,
    };

    Ok((batches, info))
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point 2 — raw tensor with explicit XYZ positions
// ─────────────────────────────────────────────────────────────────────────────

/// Load from a pre-assembled `Array2<f32>` with one **explicit** `[x,y,z]`
/// position per channel row.
///
/// The data must be raw (unprocessed) EEG in volts; the full exg pipeline is
/// applied internally.  The shape is `[n_channels, n_samples]`.
pub fn load_from_raw_tensor<B: Backend>(
    data:      Array2<f32>,
    positions: &[[f32; 3]],
    sample_rate: f32,
    data_norm:   f32,
    data_cfg:    &DataConfig,
    device:      &B::Device,
) -> anyhow::Result<Vec<InputBatch<B>>> {
    let n_ch = data.nrows();
    anyhow::ensure!(
        positions.len() == n_ch,
        "positions.len() = {} must equal data.nrows() = {}", positions.len(), n_ch
    );

    let duration_s = data.ncols() as f32 / sample_rate;
    if duration_s < 5.0 {
        bail!("recording is {duration_s:.2} s, shorter than the 5 s minimum epoch");
    }

    let pos_arr = positions_to_array(positions, n_ch);
    run_pipeline(data, pos_arr, sample_rate, data_norm, data_cfg, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point 3 — raw tensor with channel names (auto position lookup)
// ─────────────────────────────────────────────────────────────────────────────

/// Load from a pre-assembled `Array2<f32>` using **channel names** to look up
/// scalp positions from the bundled montage database.
///
/// Channels not found in any montage (e.g. custom names) get the centroid of
/// the remaining channels as their position, which keeps them encodable.
/// Pass explicit XYZ via `position_overrides` to override any channel.
pub fn load_from_named_tensor<B: Backend>(
    data:               Array2<f32>,
    channel_names:      &[&str],
    sample_rate:        f32,
    data_norm:          f32,
    position_overrides: &HashMap<String, [f32; 3]>,
    data_cfg:           &DataConfig,
    device:             &B::Device,
) -> anyhow::Result<Vec<InputBatch<B>>> {
    let n_ch = data.nrows();
    anyhow::ensure!(
        channel_names.len() == n_ch,
        "channel_names.len() = {} must equal data.nrows() = {}",
        channel_names.len(), n_ch
    );

    let duration_s = data.ncols() as f32 / sample_rate;
    if duration_s < 5.0 {
        bail!("recording is {duration_s:.2} s, shorter than the 5 s minimum epoch");
    }

    let names: Vec<String> = channel_names.iter().map(|s| s.to_string()).collect();
    let positions = resolve_positions(&names, position_overrides);
    let pos_arr   = positions_to_array(&positions, n_ch);

    run_pipeline(data, pos_arr, sample_rate, data_norm, data_cfg, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV parser (no external dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a CSV file into `(channel_names, data [C, T])`.
///
/// Rules:
/// - Lines starting with `#` are skipped.
/// - First non-blank, non-comment line is the header.
/// - The first column is the timestamp column (identified by the header name
///   containing "time" case-insensitively, or simply by being column index 0).
/// - All remaining columns are EEG channels.
fn parse_csv(path: &Path) -> anyhow::Result<(Vec<String>, Array2<f32>)> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    let mut lines = content.lines()
        .filter(|l| { let t = l.trim(); !t.is_empty() && !t.starts_with('#') });

    // ── Header ────────────────────────────────────────────────────────────────
    let header_line = lines.next()
        .ok_or_else(|| anyhow::anyhow!("CSV file is empty"))?;
    let header: Vec<&str> = header_line.split(',').collect();
    anyhow::ensure!(header.len() >= 2, "CSV must have at least a timestamp and one channel column");

    // Identify timestamp column (first column, OR first whose name ≈ "time")
    let ts_col = header.iter().position(|h| {
        let n = h.trim().to_ascii_lowercase();
        n.contains("time") || n == "t" || n == "ts"
    }).unwrap_or(0);

    // Channel names: all columns except the timestamp column
    let ch_names: Vec<String> = header.iter().enumerate()
        .filter(|&(i, _)| i != ts_col)
        .map(|(_, h)| h.trim().to_string())
        .collect();
    let n_ch = ch_names.len();
    anyhow::ensure!(n_ch >= 1, "CSV has no channel columns after timestamp");

    // ── Data rows ─────────────────────────────────────────────────────────────
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for (row_idx, line) in lines.enumerate() {
        let parts: Vec<&str> = line.split(',').collect();
        anyhow::ensure!(
            parts.len() == header.len(),
            "row {row_idx}: expected {} columns, got {}", header.len(), parts.len()
        );
        let eeg: Vec<f32> = parts.iter().enumerate()
            .filter(|&(i, _)| i != ts_col)
            .map(|(_, s)| {
                s.trim().parse::<f32>()
                    .with_context(|| format!("row {row_idx}: cannot parse '{}'", s.trim()))
            })
            .collect::<anyhow::Result<Vec<f32>>>()?;
        rows.push(eeg);
    }

    let n_t = rows.len();
    anyhow::ensure!(n_t >= 1, "CSV has no data rows");

    // ── Assemble [C, T] array ─────────────────────────────────────────────────
    // rows is currently [T, C]; transpose to [C, T]
    let mut flat = vec![0f32; n_ch * n_t];
    for (t, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            flat[c * n_t + t] = v;
        }
    }
    let data = Array2::from_shape_vec((n_ch, n_t), flat)
        .context("assembling data array")?;

    Ok((ch_names, data))
}

// ─────────────────────────────────────────────────────────────────────────────
// Position helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve XYZ positions for a list of channel names.
///
/// Priority per channel:
/// 1. `overrides` map (case-insensitive normalised key)
/// 2. [`channel_xyz`] database
/// 3. `[0.0, 0.0, 0.0]` placeholder — will be replaced by centroid after all
///    known channels are resolved.
fn resolve_positions(
    names:     &[String],
    overrides: &HashMap<String, [f32; 3]>,
) -> Vec<[f32; 3]> {
    let mut positions: Vec<[f32; 3]> = names.iter().map(|name| {
        // 1. override map
        let key = normalise(name);
        if let Some(&xyz) = overrides.iter().find(|(k, _)| normalise(k) == key).map(|(_, v)| v) {
            return xyz;
        }
        // 2. database
        if let Some(xyz) = channel_xyz(name) {
            return xyz;
        }
        // 3. placeholder
        [f32::NAN, f32::NAN, f32::NAN]
    }).collect();

    // Replace NaN placeholders with centroid of known positions
    let centroid = centroid_of(&positions);
    for p in &mut positions {
        if p[0].is_nan() { *p = centroid; }
    }

    positions
}

/// Euclidean distance between two 3-D points.
#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute centroid of non-NaN positions; returns `[0,0,0]` if none.
fn centroid_of(positions: &[[f32; 3]]) -> [f32; 3] {
    let valid: Vec<_> = positions.iter().filter(|p| !p[0].is_nan()).collect();
    if valid.is_empty() { return [0.0, 0.0, 0.0]; }
    let n = valid.len() as f32;
    let x = valid.iter().map(|p| p[0]).sum::<f32>() / n;
    let y = valid.iter().map(|p| p[1]).sum::<f32>() / n;
    let z = valid.iter().map(|p| p[2]).sum::<f32>() / n;
    [x, y, z]
}

fn positions_to_array(positions: &[[f32; 3]], n_ch: usize) -> Array2<f32> {
    let flat: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
    Array2::from_shape_vec((n_ch, 3), flat).expect("positions_to_array shape mismatch")
}

// ─────────────────────────────────────────────────────────────────────────────
// Padding
// ─────────────────────────────────────────────────────────────────────────────

/// Reorder and pad channels to match `target_channels`.
///
/// If `whitelist` is `Some`, only CSV channels whose normalised name appears
/// in the whitelist are considered "present"; others are ignored.
///
/// Returns `(padded_data [C_out, T], padded_names, padded_positions, n_padded)`.
fn apply_padding(
    data:      &Array2<f32>,
    names:     &[String],
    positions: &[[f32; 3]],
    targets:   &[String],
    strategy:  &PaddingStrategy,
    overrides: &HashMap<String, [f32; 3]>,
    whitelist: Option<&[String]>,
) -> anyhow::Result<(Array2<f32>, Vec<String>, Vec<[f32; 3]>, usize)> {
    let n_t = data.ncols();
    let mut out_rows:  Vec<Vec<f32>>   = Vec::with_capacity(targets.len());
    let mut out_names: Vec<String>     = Vec::with_capacity(targets.len());
    let mut out_pos:   Vec<[f32; 3]>   = Vec::with_capacity(targets.len());
    let mut n_padded = 0usize;

    // Build a normalised-name → source-index map for loaded channels.
    // If a whitelist is provided, only whitelisted channels count as "present".
    let wl_keys: Option<std::collections::HashSet<String>> = whitelist.map(|wl| {
        wl.iter().map(|n| normalise(n)).collect()
    });
    let src_index: HashMap<String, usize> = names.iter().enumerate()
        .filter(|(_, n)| {
            wl_keys.as_ref().map_or(true, |wl| wl.contains(&normalise(n)))
        })
        .map(|(i, n)| (normalise(n), i))
        .collect();

    // Positions of loaded channels, useful for CloneNearest.
    // Restricted to whitelisted channels when whitelist is active.
    let loaded_xyz_with_idx: Vec<([f32; 3], usize)> = positions.iter().copied()
        .enumerate()
        .filter(|(i, _)| src_index.values().any(|&si| si == *i))
        .map(|(i, xyz)| (xyz, i))
        .collect();

    for target in targets {
        let key = normalise(target);
        if let Some(&src) = src_index.get(&key) {
            // Channel present in CSV — use it as-is
            out_rows.push(data.row(src).to_vec());
            out_names.push(target.clone());
            out_pos.push(positions[src]);
        } else if matches!(strategy, PaddingStrategy::NoPadding) {
            // Drop the missing channel entirely — no synthesis, no row added.
            n_padded += 1;
            continue;
        } else {
            // Channel missing — synthesise
            n_padded += 1;

            // Position for the new channel
            let new_pos = position_for_missing(target, overrides, positions);

            let new_row = match strategy {
                PaddingStrategy::Zero => {
                    vec![0f32; n_t]
                }
                PaddingStrategy::CloneChannel(src_name) => {
                    let src_key = normalise(src_name);
                    let src_idx = src_index.get(&src_key).copied()
                        .ok_or_else(|| anyhow::anyhow!(
                            "CloneChannel source '{}' not found in CSV", src_name
                        ))?;
                    data.row(src_idx).to_vec()
                }
                PaddingStrategy::CloneNearest => {
                    // Find loaded channel whose position is closest to `new_pos`
                    let nearest_idx = nearest_channel(new_pos, &loaded_xyz_with_idx)
                        .unwrap_or(0);
                    data.row(nearest_idx).to_vec()
                }

                PaddingStrategy::InterpWeighted { k } => {
                    // Sort real channels by L2 distance, keep k nearest, then
                    // form an inverse-distance–weighted average.
                    let mut dists: Vec<(f32, usize)> = loaded_xyz_with_idx.iter()
                        .map(|&(xyz, idx)| (dist3(xyz, new_pos), idx))
                        .collect();
                    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let k_actual = (*k).min(dists.len()).max(1);
                    let k_slice  = &dists[..k_actual];
                    // weight_i = 1/d_i  (replace exact-zero distance with large weight)
                    let weights: Vec<f32> = k_slice.iter()
                        .map(|(d, _)| if *d < 1e-6 { 1e6_f32 } else { 1.0 / d })
                        .collect();
                    let w_sum: f32 = weights.iter().sum();
                    let mut interp = vec![0f32; n_t];
                    for ((_, idx), w) in k_slice.iter().zip(weights.iter()) {
                        let wn = w / w_sum;
                        for (o, &v) in interp.iter_mut().zip(data.row(*idx).iter()) {
                            *o += wn * v;
                        }
                    }
                    interp
                }

                PaddingStrategy::Mirror => {
                    // Flip the target's X coordinate to the opposite hemisphere,
                    // then find the nearest real channel to that mirror position.
                    let mirror_pos = [-new_pos[0], new_pos[1], new_pos[2]];
                    let nearest_idx = nearest_channel(mirror_pos, &loaded_xyz_with_idx)
                        .unwrap_or_else(|| loaded_xyz_with_idx.first().map(|&(_, i)| i).unwrap_or(0));
                    data.row(nearest_idx).to_vec()
                }

                PaddingStrategy::MeanRef => {
                    // Per-sample mean of all real channels.
                    let n_real = loaded_xyz_with_idx.len().max(1);
                    let mut mean_sig = vec![0f32; n_t];
                    for &(_, idx) in &loaded_xyz_with_idx {
                        for (m, &v) in mean_sig.iter_mut().zip(data.row(idx).iter()) {
                            *m += v;
                        }
                    }
                    for m in &mut mean_sig { *m /= n_real as f32; }
                    mean_sig
                }

                // Handled by the early `continue` branch above.
                PaddingStrategy::NoPadding => unreachable!(),
            };

            out_rows.push(new_row);
            out_names.push(target.clone());
            out_pos.push(new_pos);
        }
    }

    let n_out = out_rows.len();
    let flat: Vec<f32> = out_rows.into_iter().flatten().collect();
    let padded = Array2::from_shape_vec((n_out, n_t), flat)
        .context("assembling padded data array")?;

    Ok((padded, out_names, out_pos, n_padded))
}

/// Determine the XYZ position for a missing channel.
///
/// Priority: position_overrides → database lookup → centroid of existing.
fn position_for_missing(
    name:      &str,
    overrides: &HashMap<String, [f32; 3]>,
    existing:  &[[f32; 3]],
) -> [f32; 3] {
    let key = normalise(name);
    if let Some(&xyz) = overrides.iter().find(|(k, _)| normalise(k) == key).map(|(_, v)| v) {
        return xyz;
    }
    if let Some(xyz) = channel_xyz(name) {
        return xyz;
    }
    centroid_of(existing)
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared preprocessing pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full exg preprocessing pipeline and assemble `InputBatch` structs.
///
/// Pipeline (identical to [`load_from_fif`](crate::data::load_from_fif)):
/// resample → 0.5 Hz HP FIR → average reference → global z-score →
/// epoch (5 s) → baseline correction → ÷ data_norm
fn run_pipeline<B: Backend>(
    data:        Array2<f32>,    // [C, T] raw EEG in volts
    pos_arr:     Array2<f32>,    // [C, 3] metres
    sample_rate: f32,
    data_norm:   f32,
    data_cfg:    &DataConfig,
    device:      &B::Device,
) -> anyhow::Result<Vec<InputBatch<B>>> {
    use exg::PipelineConfig;

    let cfg = PipelineConfig { data_norm, ..PipelineConfig::default() };
    let epochs = exg::preprocess(data, pos_arr, sample_rate, &cfg)?;

    if epochs.is_empty() {
        bail!("recording produced zero epochs (likely shorter than the 5 s minimum epoch)");
    }

    let mut batches = Vec::with_capacity(epochs.len());
    for (eeg_arr, pos_out) in epochs {
        let (c, t) = eeg_arr.dim();
        let eeg_data: Vec<f32> = eeg_arr.iter().copied().collect();
        let eeg = Tensor::<B, 2>::from_data(TensorData::new(eeg_data, vec![c, t]), device);

        let pos_data: Vec<f32> = pos_out.iter().copied().collect();
        let chan_pos = Tensor::<B, 2>::from_data(TensorData::new(pos_data, vec![c, 3]), device);

        let chan_pos_disc = discretize_chan_pos(chan_pos.clone(), data_cfg, device);
        let tc = t / data_cfg.num_fine_time_pts;

        let (eeg_tokens, _, posd, t_coarse) =
            chop_and_reshape(eeg, chan_pos.clone(), chan_pos_disc, data_cfg.num_fine_time_pts);

        let tok_idx       = build_tok_idx(posd, t_coarse);
        let encoder_input = eeg_tokens.unsqueeze_dim::<3>(0);

        batches.push(InputBatch { encoder_input, tok_idx, chan_pos, n_channels: c, tc });
    }

    Ok(batches)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal CSV to a temp file and verify it round-trips.
    #[test]
    fn parse_csv_basic() {
        let content = "timestamp,Fp1,Fp2\n0.0,1e-5,2e-5\n0.004,3e-5,4e-5\n";
        let path = std::env::temp_dir().join("zuna_test_basic.csv");
        std::fs::write(&path, content).unwrap();
        let (names, data) = parse_csv(&path).unwrap();
        assert_eq!(names, ["Fp1", "Fp2"]);
        assert_eq!(data.dim(), (2, 2));
        assert!((data[[0, 0]] - 1e-5_f32).abs() < 1e-10);
        assert!((data[[1, 1]] - 4e-5_f32).abs() < 1e-10);
    }

    #[test]
    fn parse_csv_skips_comments() {
        let content = "# comment\ntimestamp,C3\n0.0,0.5\n0.004,-0.3\n";
        let path = std::env::temp_dir().join("zuna_test_comments.csv");
        std::fs::write(&path, content).unwrap();
        let (names, data) = parse_csv(&path).unwrap();
        assert_eq!(names, ["C3"]);
        assert_eq!(data.dim(), (1, 2));
    }

    #[test]
    fn resolve_positions_uses_database() {
        let pos = resolve_positions(&["Cz".to_string()], &HashMap::new());
        assert_eq!(pos.len(), 1);
        let [x, y, z] = pos[0];
        assert!(x.abs() < 0.12 && y.abs() < 0.12 && z.abs() < 0.12);
    }

    #[test]
    fn resolve_positions_override_wins() {
        let mut ov = HashMap::new();
        ov.insert("CZ".to_string(), [0.01, 0.02, 0.09]);
        let pos = resolve_positions(&["Cz".to_string()], &ov);
        assert_eq!(pos[0], [0.01, 0.02, 0.09]);
    }

    #[test]
    fn resolve_positions_unknown_gets_centroid() {
        let names = vec!["UNKNOWN_XYZ".to_string(), "Cz".to_string()];
        let pos = resolve_positions(&names, &HashMap::new());
        // Unknown channel should get centroid of known channels, which is Cz
        let cz = channel_xyz("Cz").unwrap();
        let centroid = pos[0]; // unknown channel
        // centroid of [unknown_placeholder, cz] → when unknown is NaN, centroid = cz
        assert!((centroid[0] - cz[0]).abs() < 1e-5);
    }

    #[test]
    fn padding_zero_adds_zero_rows() {
        let data = Array2::from_shape_vec((2, 4), vec![1f32; 8]).unwrap();
        let names = vec!["Fp1".to_string(), "Fp2".to_string()];
        let pos = resolve_positions(&names, &HashMap::new());
        let targets = vec!["Fp1".to_string(), "Fp2".to_string(), "Fz".to_string()];
        let (out, out_names, out_pos, n_padded) = apply_padding(
            &data, &names, &pos, &targets, &PaddingStrategy::Zero, &HashMap::new(), None
        ).unwrap();
        assert_eq!(out.dim(), (3, 4));
        assert_eq!(n_padded, 1);
        assert_eq!(out_names[2], "Fz");
        // Fz row must be all zeros
        assert!(out.row(2).iter().all(|&v| v == 0.0));
        // Fz must have a known position (from database)
        let [x, y, z] = out_pos[2];
        assert!(x.abs() < 0.12 && y.abs() < 0.12 && z.abs() < 0.12);
    }

    #[test]
    fn padding_clone_channel() {
        let data = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
        let names = vec!["Fp1".to_string(), "Fp2".to_string()];
        let pos = resolve_positions(&names, &HashMap::new());
        let targets = vec!["Fp1".to_string(), "Cz".to_string()];  // Cz missing
        let (out, _, _, n_padded) = apply_padding(
            &data, &names, &pos, &targets,
            &PaddingStrategy::CloneChannel("Fp1".to_string()), &HashMap::new(), None
        ).unwrap();
        assert_eq!(n_padded, 1);
        // Cz row should equal Fp1 row
        assert_eq!(out.row(0).to_vec(), out.row(1).to_vec());
    }

    #[test]
    fn padding_clone_nearest() {
        // Fp1 and Fp2 are close together; Fz is between them and Cz
        let data = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32 * 0.1).collect()).unwrap();
        let names = vec!["Fp1".to_string(), "Fp2".to_string()];
        let pos = resolve_positions(&names, &HashMap::new());
        let targets = vec!["Fp1".to_string(), "Fp2".to_string(), "AF7".to_string()];
        let (out, _, _, n_padded) = apply_padding(
            &data, &names, &pos, &targets,
            &PaddingStrategy::CloneNearest, &HashMap::new(), None
        ).unwrap();
        assert_eq!(n_padded, 1);
        // AF7 is near Fp1/Fp2 front — cloned from one of them, must be nonzero
        assert!(out.row(2).iter().any(|&v| v != 0.0));
    }
}
