/// Data preparation for ZUNA inference (burn 0.20.1)
///
/// Two entry points:
///   • `load_from_fif`  — read a raw .fif file through the exg
///                        pipeline and return `InputBatch` structs directly.
///   • `load_batch`     — read a pre-exported safetensors batch
///                        (legacy / Python-compatible path).
use burn::prelude::*;
use safetensors::SafeTensors;
use crate::config::DataConfig;

// ── 1. Discretise channel positions ─────────────────────────────────────────

/// Map continuous scalp xyz positions to integer bin indices [0, num_bins-1].
/// Python equivalent: `discretize_chan_pos` in eeg_data.py.
///
/// chan_pos: [C, 3], cfg.xyz_min/max in metres, cfg.num_bins = 50.
pub fn discretize_chan_pos<B: Backend>(
    chan_pos: Tensor<B, 2>,
    cfg:      &DataConfig,
    device:   &B::Device,
) -> Tensor<B, 2, Int> {
    let [_c, _] = chan_pos.dims();
    let xyz_min = Tensor::<B, 2>::from_data(
        TensorData::new(cfg.xyz_min.to_vec(), vec![1, 3]), device,
    );
    let xyz_max = Tensor::<B, 2>::from_data(
        TensorData::new(cfg.xyz_max.to_vec(), vec![1, 3]), device,
    );

    let norm = (chan_pos - xyz_min.clone()) / (xyz_max - xyz_min); // [C, 3] in [0,1]
    let bins = cfg.num_bins as f32;
    norm.mul_scalar(bins)
        .int()
        .clamp(0i32, cfg.num_bins as i32 - 1)
}

// ── 2. Chop-and-reshape (mode "B") ──────────────────────────────────────────

/// Reshape [C, T] → [C*tc, tf] token matrix.
/// Python: `chop_and_reshape_signals(..., use_coarse_time="B")`.
///
/// Returns (eeg_tokens [C*tc, tf], chan_pos_rep [C*tc, 3],
///          chan_pos_disc_rep [C*tc, 3], t_coarse [C*tc, 1])
pub fn chop_and_reshape<B: Backend>(
    eeg:          Tensor<B, 2>,       // [C, T]
    chan_pos:     Tensor<B, 2>,       // [C, 3]
    chan_pos_disc: Tensor<B, 2, Int>, // [C, 3]
    tf:           usize,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let [c, t_total] = eeg.dims();
    assert_eq!(t_total % tf, 0, "T must be divisible by tf");
    let tc = t_total / tf;
    let s  = c * tc;
    let device = eeg.device();

    // [C, T] → [C, tc, tf] → [C*tc, tf]
    let eeg_tokens = eeg.reshape([c, tc, tf]).reshape([s, tf]);

    // Repeat each channel position tc times: [C, 3] → [C*tc, 3]
    let pos  = repeat_interleave_rows_f(chan_pos,       tc);
    let posd = repeat_interleave_rows_i(chan_pos_disc,  tc);

    // t_coarse: [0,1,...,tc-1] repeated C times → [C*tc, 1]
    let tc_vals: Vec<i32> = (0..tc as i32)
        .cycle()
        .take(s)
        .collect();
    let t_coarse = Tensor::<B, 1, Int>::from_data(
        TensorData::new(tc_vals, vec![s]),
        &device,
    )
    .reshape([s, 1]);

    (eeg_tokens, pos, posd, t_coarse)
}

// ── 3. Build token index [S, 4] ──────────────────────────────────────────────

/// Concatenate discrete (x,y,z) and t_coarse → [S, 4].
/// Python: `cat((chan_pos_discrete, t_coarse), dim=2)` (we drop the batch dim).
pub fn build_tok_idx<B: Backend>(
    chan_pos_disc: Tensor<B, 2, Int>,  // [S, 3]
    t_coarse:     Tensor<B, 2, Int>,  // [S, 1]
) -> Tensor<B, 2, Int> {
    Tensor::cat(vec![chan_pos_disc, t_coarse], 1)  // [S, 4]
}

// ── 4. InputBatch ─────────────────────────────────────────────────────────────

pub struct InputBatch<B: Backend> {
    /// [1, S, tf]  — encoder input (normalised, zeroed = dropped channel)
    pub encoder_input: Tensor<B, 3>,
    /// [S, 4]  — 4-D RoPE token indices
    pub tok_idx: Tensor<B, 2, Int>,
    /// [C, 3]  — continuous channel positions (metres)
    pub chan_pos: Tensor<B, 2>,
    pub n_channels: usize,
    pub tc: usize,
}

// ── 5. Load a safetensors batch file ─────────────────────────────────────────

/// Load a safetensors file created by `scripts/export_batch.py`.
///
/// Expected keys:
///   `n_samples`       int32 scalar
///   `eeg_{i}`         float32 [C, T]  (already /data_norm)
///   `chan_pos_{i}`    float32 [C, 3]
pub fn load_batch<B: Backend>(
    path:   &str,
    cfg:    &DataConfig,
    device: &B::Device,
) -> anyhow::Result<Vec<InputBatch<B>>> {
    let bytes = std::fs::read(path)?;
    let st    = SafeTensors::deserialize(&bytes)?;

    let n_samples = {
        let v = st.tensor("n_samples")?;
        match v.dtype() {
            // preprocess_fif.py writes I32; infer binary writes F32
            safetensors::Dtype::I32 => {
                let b: [u8; 4] = v.data().get(..4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| anyhow::anyhow!("n_samples I32 too short"))?;
                i32::from_le_bytes(b) as usize
            }
            safetensors::Dtype::F32 => {
                let b: [u8; 4] = v.data().get(..4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| anyhow::anyhow!("n_samples F32 too short"))?;
                f32::from_le_bytes(b) as usize
            }
            other => anyhow::bail!("unexpected dtype for n_samples: {:?}", other),
        }
    };

    let mut batches = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // EEG signal [C, T]
        let eeg_view = st.tensor(&format!("eeg_{i}"))?;
        let [c, t]: [usize; 2] = eeg_view.shape().try_into()
            .map_err(|_| anyhow::anyhow!("eeg_{i} must be 2-D"))?;
        let eeg_f32 = bytes_to_f32(eeg_view.data(), eeg_view.dtype())?;
        let eeg = Tensor::<B, 2>::from_data(TensorData::new(eeg_f32, vec![c, t]), device);

        // Channel positions [C, 3]
        let pos_view = st.tensor(&format!("chan_pos_{i}"))?;
        let pos_f32  = bytes_to_f32(pos_view.data(), pos_view.dtype())?;
        let chan_pos = Tensor::<B, 2>::from_data(TensorData::new(pos_f32, vec![c, 3]), device);

        let chan_pos_disc = discretize_chan_pos(chan_pos.clone(), cfg, device);
        let tc = t / cfg.num_fine_time_pts;

        let (eeg_tokens, _, posd, t_coarse) =
            chop_and_reshape(eeg.clone(), chan_pos.clone(), chan_pos_disc, cfg.num_fine_time_pts);

        let tok_idx     = build_tok_idx(posd, t_coarse);
        let encoder_input = eeg_tokens.unsqueeze_dim::<3>(0); // [1, S, tf]

        batches.push(InputBatch { encoder_input, tok_idx, chan_pos, n_channels: c, tc });
    }

    Ok(batches)
}

// ── 6. Invert reshape ─────────────────────────────────────────────────────────

/// [C*tc, tf] → [C, T]  (inverse of `chop_and_reshape` mode "B")
pub fn invert_reshape<B: Backend>(
    tokens:     Tensor<B, 2>,
    n_channels: usize,
    tc:         usize,
    tf:         usize,
) -> Tensor<B, 2> {
    tokens.reshape([n_channels, tc, tf]).reshape([n_channels, tc * tf])
}

// ── 7. FIF metadata (for verbose printing) ───────────────────────────────────

/// Metadata extracted from a FIF file header — returned alongside batches.
pub struct FifInfo {
    /// Channel names in order.
    pub ch_names: Vec<String>,
    /// Scalp positions in **millimetres** `[C, 3]` (x=right, y=anterior, z=superior).
    pub ch_pos_mm: Vec<[f32; 3]>,
    /// Original sampling rate (Hz).
    pub sfreq: f32,
    /// Number of time points in the raw file (before resampling).
    pub n_times_raw: usize,
    /// Duration in seconds.
    pub duration_s: f32,
    /// Number of epochs produced by the pipeline.
    pub n_epochs: usize,
    /// Target sfreq used by the pipeline.
    pub target_sfreq: f32,
    /// Epoch duration (s).
    pub epoch_dur_s: f32,
}

// ── 8. Load directly from a FIF file ─────────────────────────────────────────

/// Preprocess a `.fif` file through the exg pipeline and return
/// ready-to-run `InputBatch` structs plus metadata — no Python required.
///
/// Pipeline applied (matches `preprocess_fif.py`):
///   resample → 0.5 Hz highpass FIR → average reference → global z-score
///   → epoch (5 s @ 256 Hz = 1280 pts) → ÷ data_norm
///
/// Channel positions are read from the FIF `ch_info.loc[0..3]` (metres).
pub fn load_from_fif<B: Backend>(
    path:      &std::path::Path,
    data_cfg:  &DataConfig,
    data_norm: f32,
    device:    &B::Device,
) -> anyhow::Result<(Vec<InputBatch<B>>, FifInfo)> {
    use exg::{
        fiff::raw::open_raw,
        PipelineConfig,
    };
    use ndarray::Array2;

    // ── 1. Open FIF ─────────────────────────────────────────────────────────
    let raw_fif      = open_raw(path)?;
    let src_sfreq    = raw_fif.info.sfreq as f32;
    let n_ch         = raw_fif.info.n_chan;
    let n_times_raw  = raw_fif.n_times();
    let duration_s   = n_times_raw as f32 / src_sfreq;

    // ── 2. Channel names & positions ────────────────────────────────────────
    let ch_names: Vec<String> = raw_fif.info.chs.iter()
        .map(|ch| ch.name.clone())
        .collect();
    let ch_pos_mm: Vec<[f32; 3]> = raw_fif.info.chs.iter()
        .map(|ch| [ch.loc[0] * 1000.0, ch.loc[1] * 1000.0, ch.loc[2] * 1000.0])
        .collect();

    let pos_flat: Vec<f32> = raw_fif.info.chs.iter()
        .flat_map(|ch| [ch.loc[0], ch.loc[1], ch.loc[2]])
        .collect();
    let chan_pos_arr = Array2::from_shape_vec((n_ch, 3), pos_flat)?;

    // ── 3. Read raw data [C, T] ─────────────────────────────────────────────
    let data_f64  = raw_fif.read_all_data()?;
    let data_f32: Array2<f32> = data_f64.mapv(|v| v as f32);

    // ── 4. Preprocessing pipeline ───────────────────────────────────────────
    let preproc_cfg = PipelineConfig {
        data_norm,
        ..PipelineConfig::default()
    };

    let epochs = exg::preprocess(data_f32, chan_pos_arr, src_sfreq, &preproc_cfg)?;
    let n_epochs = epochs.len();

    // ── 5. Convert each epoch to InputBatch<B> ──────────────────────────────
    let mut batches = Vec::with_capacity(n_epochs);

    for (eeg_arr, pos_arr) in epochs {
        let (c, t) = eeg_arr.dim();

        let eeg_data: Vec<f32> = eeg_arr.iter().copied().collect();
        let eeg = Tensor::<B, 2>::from_data(TensorData::new(eeg_data, vec![c, t]), device);

        let pos_data: Vec<f32> = pos_arr.iter().copied().collect();
        let chan_pos_t = Tensor::<B, 2>::from_data(TensorData::new(pos_data, vec![c, 3]), device);

        let chan_pos_disc = discretize_chan_pos(chan_pos_t.clone(), data_cfg, device);
        let tc = t / data_cfg.num_fine_time_pts;

        let (eeg_tokens, _, posd, t_coarse) = chop_and_reshape(
            eeg,
            chan_pos_t.clone(),
            chan_pos_disc,
            data_cfg.num_fine_time_pts,
        );

        let tok_idx       = build_tok_idx(posd, t_coarse);
        let encoder_input = eeg_tokens.unsqueeze_dim::<3>(0); // [1, S, tf]

        batches.push(InputBatch {
            encoder_input,
            tok_idx,
            chan_pos: chan_pos_t,
            n_channels: c,
            tc,
        });
    }

    let info = FifInfo {
        ch_names,
        ch_pos_mm,
        sfreq: src_sfreq,
        n_times_raw,
        duration_s,
        n_epochs,
        target_sfreq: preproc_cfg.target_sfreq,
        epoch_dur_s:  preproc_cfg.epoch_dur,
    };

    Ok((batches, info))
}

// ── 9. CPU-only FIF preprocessing (no burn tensors — Send-safe) ──────────────

/// Preprocessed epoch data in plain Vecs (no burn tensors).
/// Safe to produce on any thread and convert to tensors later.
pub struct PreprocessedEpoch {
    /// EEG token data, row-major `[S, tf]` float32.
    pub eeg_tokens: Vec<f32>,
    /// Token indices, row-major `[S, 4]` int32.
    pub tok_idx: Vec<i32>,
    /// Channel positions in metres, row-major `[C, 3]` float32.
    pub chan_pos: Vec<f32>,
    /// Number of tokens S = n_channels × tc.
    pub s: usize,
    /// Fine time points per token.
    pub tf: usize,
    /// Number of EEG channels.
    pub n_channels: usize,
    /// Coarse time steps per epoch.
    pub tc: usize,
}

/// Result of CPU-only FIF preprocessing.
pub struct PreprocessedFif {
    pub epochs: Vec<PreprocessedEpoch>,
    pub info: FifInfo,
}

/// Preprocess a FIF file entirely on the CPU without creating burn tensors.
///
/// This is the parallel-safe counterpart of [`load_from_fif`]: it returns
/// plain `Vec<f32>` / `Vec<i32>` buffers that can be produced on a rayon
/// worker and later converted to tensors on the main thread.
pub fn preprocess_fif_cpu(
    path:      &std::path::Path,
    data_cfg:  &DataConfig,
    data_norm: f32,
) -> anyhow::Result<PreprocessedFif> {
    use exg::{fiff::raw::open_raw, PipelineConfig};
    use ndarray::Array2;

    let raw_fif     = open_raw(path)?;
    let src_sfreq   = raw_fif.info.sfreq as f32;
    let n_ch        = raw_fif.info.n_chan;
    let n_times_raw = raw_fif.n_times();
    let duration_s  = n_times_raw as f32 / src_sfreq;

    let ch_names: Vec<String> = raw_fif.info.chs.iter().map(|ch| ch.name.clone()).collect();
    let ch_pos_mm: Vec<[f32; 3]> = raw_fif.info.chs.iter()
        .map(|ch| [ch.loc[0] * 1000.0, ch.loc[1] * 1000.0, ch.loc[2] * 1000.0])
        .collect();
    let pos_flat: Vec<f32> = raw_fif.info.chs.iter()
        .flat_map(|ch| [ch.loc[0], ch.loc[1], ch.loc[2]])
        .collect();
    let chan_pos_arr = Array2::from_shape_vec((n_ch, 3), pos_flat)?;

    let data_f64 = raw_fif.read_all_data()?;
    let data_f32: Array2<f32> = data_f64.mapv(|v| v as f32);

    let preproc_cfg = PipelineConfig { data_norm, ..PipelineConfig::default() };
    let exg_epochs = exg::preprocess(data_f32, chan_pos_arr, src_sfreq, &preproc_cfg)?;
    let n_epochs = exg_epochs.len();

    // Discretize + chop using temporary ndarray math (no burn tensors).
    let tf = data_cfg.num_fine_time_pts;
    let mut epochs = Vec::with_capacity(n_epochs);

    for (eeg_arr, pos_arr) in exg_epochs {
        let (c, t) = eeg_arr.dim();
        let tc = t / tf;

        // Discretize channel positions to bins [0, num_bins-1].
        let bins = data_cfg.num_bins as f32;
        let disc: Vec<i32> = pos_arr.iter().enumerate().map(|(i, &v)| {
            let axis = i % 3;
            let lo = data_cfg.xyz_min[axis];
            let hi = data_cfg.xyz_max[axis];
            let norm = (v - lo) / (hi - lo);
            (norm * bins).min(bins - 1.0).max(0.0) as i32
        }).collect();

        // Chop and reshape: [C, T] → [C*tc, tf]
        // Also build tok_idx [S, 4] = [disc_x, disc_y, disc_z, t_coarse]
        let s = c * tc;
        let mut eeg_tokens = vec![0f32; s * tf];
        let mut tok_idx = vec![0i32; s * 4];

        for ch in 0..c {
            for ti in 0..tc {
                let token = ch * tc + ti;
                // Copy tf samples from eeg_arr[ch, ti*tf .. (ti+1)*tf]
                for f in 0..tf {
                    eeg_tokens[token * tf + f] = eeg_arr[[ch, ti * tf + f]];
                }
                // tok_idx = [x_bin, y_bin, z_bin, t_coarse]
                tok_idx[token * 4]     = disc[ch * 3];
                tok_idx[token * 4 + 1] = disc[ch * 3 + 1];
                tok_idx[token * 4 + 2] = disc[ch * 3 + 2];
                tok_idx[token * 4 + 3] = ti as i32;
            }
        }

        let chan_pos: Vec<f32> = pos_arr.iter().copied().collect();

        epochs.push(PreprocessedEpoch { eeg_tokens, tok_idx, chan_pos, s, tf, n_channels: c, tc });
    }

    let info = FifInfo {
        ch_names, ch_pos_mm, sfreq: src_sfreq, n_times_raw, duration_s,
        n_epochs, target_sfreq: preproc_cfg.target_sfreq, epoch_dur_s: preproc_cfg.epoch_dur,
    };

    Ok(PreprocessedFif { epochs, info })
}

/// Convert a [`PreprocessedEpoch`] to a burn [`InputBatch`] on the given device.
pub fn preprocessed_to_batch<B: Backend>(
    ep:     PreprocessedEpoch,
    device: &B::Device,
) -> InputBatch<B> {
    let s  = ep.s;
    let tf = ep.tf;
    let c  = ep.n_channels;

    let encoder_input = Tensor::<B, 2>::from_data(
        TensorData::new(ep.eeg_tokens, vec![s, tf]), device,
    ).unsqueeze_dim::<3>(0); // [1, S, tf]

    let tok_idx = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ep.tok_idx, vec![s, 4]), device,
    );

    let chan_pos = Tensor::<B, 2>::from_data(
        TensorData::new(ep.chan_pos, vec![c, 3]), device,
    );

    InputBatch { encoder_input, tok_idx, chan_pos, n_channels: c, tc: ep.tc }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn repeat_interleave_rows_f<B: Backend>(t: Tensor<B, 2>, repeats: usize) -> Tensor<B, 2> {
    let [s, c] = t.dims();
    t.unsqueeze_dim::<3>(1).expand([s, repeats, c]).reshape([s * repeats, c])
}

fn repeat_interleave_rows_i<B: Backend>(
    t: Tensor<B, 2, Int>,
    repeats: usize,
) -> Tensor<B, 2, Int> {
    let [s, c] = t.dims();
    t.unsqueeze_dim::<3>(1).expand([s, repeats, c]).reshape([s * repeats, c])
}

fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype) -> anyhow::Result<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 =>
            Ok(data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
        safetensors::Dtype::BF16 =>
            Ok(data.chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
        other => anyhow::bail!("unsupported dtype {:?}", other),
    }
}
