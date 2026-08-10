//! Data preparation for ZUNA inference.
//!
//! CPU preprocessing ([`preprocess_fif_cpu`]) is always available. Burn
//! tensor helpers (`InputBatch`, `load_from_fif`, …) require
//! `--features burn`.

use std::path::Path;

use crate::config::DataConfig;

// ── Shared CPU types (Send-safe, used by RLX and Burn) ─────────────────────

/// Metadata extracted from a FIF file header.
pub struct FifInfo {
    pub ch_names: Vec<String>,
    pub ch_pos_mm: Vec<[f32; 3]>,
    pub sfreq: f32,
    pub n_times_raw: usize,
    pub duration_s: f32,
    pub n_epochs: usize,
    pub target_sfreq: f32,
    pub epoch_dur_s: f32,
}

/// One preprocessed 5-second EEG epoch (plain buffers).
pub struct PreprocessedEpoch {
    pub eeg_tokens: Vec<f32>,
    pub tok_idx: Vec<i32>,
    pub chan_pos: Vec<f32>,
    pub s: usize,
    pub tf: usize,
    pub n_channels: usize,
    pub tc: usize,
}

pub struct PreprocessedFif {
    pub epochs: Vec<PreprocessedEpoch>,
    pub info: FifInfo,
}

pub fn preprocess_fif_cpu(
    path: &Path,
    data_cfg: &DataConfig,
    data_norm: f32,
) -> anyhow::Result<PreprocessedFif> {
    use exg::{fiff::raw::open_raw, PipelineConfig};
    use ndarray::Array2;

    let raw_fif = open_raw(path)?;
    let src_sfreq = raw_fif.info.sfreq as f32;
    let n_ch = raw_fif.info.n_chan;
    let n_times_raw = raw_fif.n_times();
    let duration_s = n_times_raw as f32 / src_sfreq;

    let ch_names: Vec<String> = raw_fif.info.chs.iter().map(|ch| ch.name.clone()).collect();
    let ch_pos_mm: Vec<[f32; 3]> = raw_fif
        .info
        .chs
        .iter()
        .map(|ch| [ch.loc[0] * 1000.0, ch.loc[1] * 1000.0, ch.loc[2] * 1000.0])
        .collect();
    let pos_flat: Vec<f32> = raw_fif
        .info
        .chs
        .iter()
        .flat_map(|ch| [ch.loc[0], ch.loc[1], ch.loc[2]])
        .collect();
    let chan_pos_arr = Array2::from_shape_vec((n_ch, 3), pos_flat)?;

    let data_f64 = raw_fif.read_all_data()?;
    let data_f32: Array2<f32> = data_f64.mapv(|v| v as f32);

    let preproc_cfg = PipelineConfig {
        data_norm,
        ..PipelineConfig::default()
    };
    let exg_epochs = exg::preprocess(data_f32, chan_pos_arr, src_sfreq, &preproc_cfg)?;
    let n_epochs = exg_epochs.len();
    let tf = data_cfg.num_fine_time_pts;
    let mut epochs = Vec::with_capacity(n_epochs);

    for (eeg_arr, pos_arr) in exg_epochs {
        let (c, t) = eeg_arr.dim();
        let tc = t / tf;
        let bins = data_cfg.num_bins as f32;
        let disc: Vec<i32> = pos_arr
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let axis = i % 3;
                let lo = data_cfg.xyz_min[axis];
                let hi = data_cfg.xyz_max[axis];
                let norm = (v - lo) / (hi - lo);
                (norm * bins).min(bins - 1.0).max(0.0) as i32
            })
            .collect();

        let s = c * tc;
        let mut eeg_tokens = vec![0f32; s * tf];
        let mut tok_idx = vec![0i32; s * 4];
        for ch in 0..c {
            for ti in 0..tc {
                let token = ch * tc + ti;
                for f in 0..tf {
                    eeg_tokens[token * tf + f] = eeg_arr[[ch, ti * tf + f]];
                }
                tok_idx[token * 4] = disc[ch * 3];
                tok_idx[token * 4 + 1] = disc[ch * 3 + 1];
                tok_idx[token * 4 + 2] = disc[ch * 3 + 2];
                tok_idx[token * 4 + 3] = ti as i32;
            }
        }
        let chan_pos: Vec<f32> = pos_arr.iter().copied().collect();
        epochs.push(PreprocessedEpoch {
            eeg_tokens,
            tok_idx,
            chan_pos,
            s,
            tf,
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
        epoch_dur_s: preproc_cfg.epoch_dur,
    };
    Ok(PreprocessedFif { epochs, info })
}

pub fn invert_reshape(tokens: &[f32], n_channels: usize, tc: usize, tf: usize) -> Vec<f32> {
    assert_eq!(tokens.len(), n_channels * tc * tf);
    let mut out = vec![0f32; n_channels * tc * tf];
    for ch in 0..n_channels {
        for ti in 0..tc {
            for f in 0..tf {
                out[ch * (tc * tf) + ti * tf + f] = tokens[(ch * tc + ti) * tf + f];
            }
        }
    }
    out
}

// ── Burn tensor pipeline ─────────────────────────────────────────────────────

#[cfg(feature = "burn")]
mod burn_data {
    use super::*;
    use burn::prelude::*;
    use safetensors::SafeTensors;

    pub fn discretize_chan_pos<B: Backend>(
        chan_pos: Tensor<B, 2>,
        cfg: &DataConfig,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        let [_c, _] = chan_pos.dims();
        let xyz_min =
            Tensor::<B, 2>::from_data(TensorData::new(cfg.xyz_min.to_vec(), vec![1, 3]), device);
        let xyz_max =
            Tensor::<B, 2>::from_data(TensorData::new(cfg.xyz_max.to_vec(), vec![1, 3]), device);
        let norm = (chan_pos - xyz_min.clone()) / (xyz_max - xyz_min);
        let bins = cfg.num_bins as f32;
        norm.mul_scalar(bins)
            .int()
            .clamp(0i32, cfg.num_bins as i32 - 1)
    }

    /// `(eeg_tokens, chan_pos, chan_pos_discrete, t_coarse)` — the result of
    /// [`chop_and_reshape`].
    pub type ChoppedTokens<B> = (
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2, Int>,
        Tensor<B, 2, Int>,
    );

    pub fn chop_and_reshape<B: Backend>(
        eeg: Tensor<B, 2>,
        chan_pos: Tensor<B, 2>,
        chan_pos_disc: Tensor<B, 2, Int>,
        tf: usize,
    ) -> ChoppedTokens<B> {
        let [c, t_total] = eeg.dims();
        assert_eq!(t_total % tf, 0);
        let tc = t_total / tf;
        let s = c * tc;
        let device = eeg.device();
        let eeg_tokens = eeg.reshape([c, tc, tf]).reshape([s, tf]);
        let pos = repeat_interleave_rows_f(chan_pos, tc);
        let posd = repeat_interleave_rows_i(chan_pos_disc, tc);
        let tc_vals: Vec<i32> = (0..tc as i32).cycle().take(s).collect();
        let t_coarse = Tensor::<B, 1, Int>::from_data(TensorData::new(tc_vals, vec![s]), &device)
            .reshape([s, 1]);
        (eeg_tokens, pos, posd, t_coarse)
    }

    pub fn build_tok_idx<B: Backend>(
        chan_pos_disc: Tensor<B, 2, Int>,
        t_coarse: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2, Int> {
        Tensor::cat(vec![chan_pos_disc, t_coarse], 1)
    }

    pub struct InputBatch<B: Backend> {
        pub encoder_input: Tensor<B, 3>,
        pub tok_idx: Tensor<B, 2, Int>,
        pub chan_pos: Tensor<B, 2>,
        pub n_channels: usize,
        pub tc: usize,
    }

    pub fn load_batch<B: Backend>(
        path: &Path,
        _data_cfg: &DataConfig,
        device: &B::Device,
    ) -> anyhow::Result<Vec<InputBatch<B>>> {
        let bytes = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&bytes)?;
        let mut batches = Vec::new();
        let mut i = 0;
        loop {
            let prefix = format!("encoder_input_{i}");
            let Some(tensor) = st.tensor(&prefix).ok() else {
                break;
            };
            let shape: Vec<usize> = tensor.shape().to_vec();
            let data = bytes_to_f32(tensor.data(), tensor.dtype())?;
            let encoder_input = Tensor::<B, 3>::from_data(TensorData::new(data, shape), device);
            let tok_idx_t = st.tensor(&format!("tok_idx_{i}"))?;
            let tok_shape: Vec<usize> = tok_idx_t.shape().to_vec();
            let tok_data: Vec<i32> = tok_idx_t
                .data()
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tok_idx =
                Tensor::<B, 2, Int>::from_data(TensorData::new(tok_data, tok_shape), device);
            let pos_t = st.tensor(&format!("chan_pos_{i}"))?;
            let pos_shape: Vec<usize> = pos_t.shape().to_vec();
            let pos_data = bytes_to_f32(pos_t.data(), pos_t.dtype())?;
            let chan_pos = Tensor::<B, 2>::from_data(TensorData::new(pos_data, pos_shape), device);
            let n_channels = chan_pos.dims()[0];
            let s = tok_idx.dims()[0];
            let tc = s / n_channels;
            batches.push(InputBatch {
                encoder_input,
                tok_idx,
                chan_pos,
                n_channels,
                tc,
            });
            i += 1;
        }
        Ok(batches)
    }

    pub fn load_from_fif<B: Backend>(
        path: &Path,
        data_cfg: &DataConfig,
        data_norm: f32,
        device: &B::Device,
    ) -> anyhow::Result<(Vec<InputBatch<B>>, FifInfo)> {
        let pre = preprocess_fif_cpu(path, data_cfg, data_norm)?;
        let mut batches = Vec::with_capacity(pre.epochs.len());
        for ep in pre.epochs {
            batches.push(preprocessed_to_batch(ep, device));
        }
        Ok((batches, pre.info))
    }

    pub fn invert_reshape<B: Backend>(
        tokens: Tensor<B, 2>,
        n_channels: usize,
        tc: usize,
        tf: usize,
    ) -> Tensor<B, 2> {
        tokens
            .reshape([n_channels, tc, tf])
            .reshape([n_channels, tc * tf])
    }

    pub fn preprocessed_to_batch<B: Backend>(
        ep: PreprocessedEpoch,
        device: &B::Device,
    ) -> InputBatch<B> {
        let s = ep.s;
        let tf = ep.tf;
        let c = ep.n_channels;
        let encoder_input =
            Tensor::<B, 2>::from_data(TensorData::new(ep.eeg_tokens, vec![s, tf]), device)
                .unsqueeze_dim::<3>(0);
        let tok_idx =
            Tensor::<B, 2, Int>::from_data(TensorData::new(ep.tok_idx, vec![s, 4]), device);
        let chan_pos = Tensor::<B, 2>::from_data(TensorData::new(ep.chan_pos, vec![c, 3]), device);
        InputBatch {
            encoder_input,
            tok_idx,
            chan_pos,
            n_channels: c,
            tc: ep.tc,
        }
    }

    fn repeat_interleave_rows_f<B: Backend>(t: Tensor<B, 2>, repeats: usize) -> Tensor<B, 2> {
        let [s, c] = t.dims();
        t.unsqueeze_dim::<3>(1)
            .expand([s, repeats, c])
            .reshape([s * repeats, c])
    }

    fn repeat_interleave_rows_i<B: Backend>(
        t: Tensor<B, 2, Int>,
        repeats: usize,
    ) -> Tensor<B, 2, Int> {
        let [s, c] = t.dims();
        t.unsqueeze_dim::<3>(1)
            .expand([s, repeats, c])
            .reshape([s * repeats, c])
    }

    fn bytes_to_f32(data: &[u8], dtype: safetensors::Dtype) -> anyhow::Result<Vec<f32>> {
        match dtype {
            safetensors::Dtype::F32 => Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            safetensors::Dtype::BF16 => Ok(data
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            other => anyhow::bail!("unsupported dtype {:?}", other),
        }
    }
}

#[cfg(feature = "burn")]
pub use burn_data::{
    build_tok_idx, chop_and_reshape, discretize_chan_pos, invert_reshape as invert_reshape_tensor,
    load_batch, load_from_fif, preprocessed_to_batch, InputBatch,
};
