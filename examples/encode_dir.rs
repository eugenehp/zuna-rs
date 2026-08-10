//! Batch-encode every `.fif` in a directory through the RLX ZUNA encoder
//! and save a per-subject safetensors file.
//!
//! Designed for the maet-rs pipeline (Option B / per-trial token slice):
//! each subject's FIF contains all of its 1-second trials concatenated as
//! continuous data at sfreq=500 Hz. ZUNA's preprocessor resamples to 256 Hz
//! and chunks into 5 s = 1280-sample epochs (40 tokens × 32 fine-time-pts
//! per channel). Five of our original 1-second trials fit into each ZUNA
//! epoch, occupying 8 contiguous tokens per channel each.
//!
//! Output per subject `sub<N>ar.fif` → `sub<N>ar.zuna.safetensors` matches
//! the safetensors layout of `crate::EncodingResult::save_safetensors`:
//!   - `embeddings_E` — `[n_tokens, output_dim]` f32
//!   - `tok_idx_E`    — `[n_tokens, 4]` i64
//!   - `chan_pos_E`   — `[n_channels, 3]` f32
//!   - `n_samples`    — scalar f32 (number of epochs)
//!
//! Usage:
//! ```sh
//! cargo run --release --example encode_dir --features "rlx,rlx-cpu" -- \
//!     --in /Users/Shared/maet-rs/data/fif \
//!     --out /Users/Shared/maet-rs/data/zuna \
//!     --device cpu
//! ```

#[path = "common/mod.rs"]
mod common;

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use safetensors::{Dtype, View};

use zuna_rs::rlx::ZunaEncoder;
use zuna_rs::{preprocess_fif_cpu, DataConfig};

#[derive(Parser, Debug)]
#[command(
    name = "encode_dir",
    about = "ZUNA encoder over a directory of FIFs (RLX backend)"
)]
struct Args {
    #[arg(long)]
    in_dir: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    /// rlx Device — cpu | metal | mlx | gpu | cuda | rocm
    #[arg(long, default_value = "cpu")]
    device: String,
    /// ZUNA training data norm divisor.
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,
    #[arg(long, default_value = common::DEFAULT_REPO, env = "ZUNA_REPO")]
    repo: String,
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<String>,
    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<String>,
    #[arg(long, env = "HF_HOME")]
    hf_cache: Option<PathBuf>,
}

fn parse_device(s: &str) -> anyhow::Result<rlx::Device> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "cpu" => rlx::Device::Cpu,
        "metal" => rlx::Device::Metal,
        "mlx" => rlx::Device::Mlx,
        "gpu" => rlx::Device::Gpu,
        "cuda" => rlx::Device::Cuda,
        "rocm" => rlx::Device::Rocm,
        other => anyhow::bail!("unknown device: {other}"),
    })
}

// ── safetensors helpers ──────────────────────────────────────────────────────

struct Owned {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: Dtype,
}
impl View for Owned {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
fn i32_to_i64_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for &x in v {
        out.extend_from_slice(&(x as i64).to_le_bytes());
    }
    out
}

fn encode_one_fif(
    enc: &mut ZunaEncoder,
    fif: &Path,
    out: &Path,
    data_cfg: &DataConfig,
    data_norm: f32,
) -> anyhow::Result<(usize, usize, usize, f64, f64)> {
    let t_pp = Instant::now();
    let pfif = preprocess_fif_cpu(fif, data_cfg, data_norm)?;
    let ms_pp = t_pp.elapsed().as_secs_f64() * 1000.0;

    let t_enc = Instant::now();
    let mut tensors: Vec<(String, Owned)> = Vec::with_capacity(pfif.epochs.len() * 3 + 1);
    let mut n_channels = 0;
    let mut tc = 0;
    let mut output_dim = 0;
    let n_ep = pfif.epochs.len();
    for (i, ep) in pfif.epochs.iter().enumerate() {
        let emb = enc.encode_one(
            &ep.eeg_tokens,
            &ep.tok_idx,
            &ep.chan_pos,
            ep.n_channels,
            ep.tc,
        )?;
        n_channels = ep.n_channels;
        tc = ep.tc;
        output_dim = emb.shape[1];

        let n_tokens = emb.shape[0];
        tensors.push((
            format!("embeddings_{i}"),
            Owned {
                data: f32_bytes(&emb.embeddings),
                shape: emb.shape.clone(),
                dtype: Dtype::F32,
            },
        ));
        tensors.push((
            format!("tok_idx_{i}"),
            Owned {
                data: i32_to_i64_bytes(&emb.tok_idx),
                shape: vec![n_tokens, 4],
                dtype: Dtype::I64,
            },
        ));
        tensors.push((
            format!("chan_pos_{i}"),
            Owned {
                data: f32_bytes(&emb.chan_pos),
                shape: vec![ep.n_channels, 3],
                dtype: Dtype::F32,
            },
        ));
    }
    tensors.push((
        "n_samples".to_string(),
        Owned {
            data: f32_bytes(&[n_ep as f32]),
            shape: vec![1],
            dtype: Dtype::F32,
        },
    ));
    let pairs: Vec<(&str, Owned)> = tensors
        .iter()
        .map(|(k, v)| {
            (
                k.as_str(),
                Owned {
                    data: v.data.clone(),
                    shape: v.shape.clone(),
                    dtype: v.dtype,
                },
            )
        })
        .collect();
    let bytes = safetensors::serialize(pairs, None)?;
    std::fs::write(out, bytes)?;
    let ms_enc = t_enc.elapsed().as_secs_f64() * 1000.0;
    Ok((n_ep, n_channels * tc, output_dim, ms_pp, ms_enc))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir)?;

    let (weights_path, config_path) = common::resolve_weights(
        &args.repo,
        args.weights.as_deref(),
        args.config.as_deref(),
        args.hf_cache.as_deref(),
    )?;
    eprintln!("[encode_dir] weights = {}", weights_path.display());
    eprintln!("[encode_dir] config  = {}", config_path.display());

    let device = parse_device(&args.device)?;
    eprintln!("[encode_dir] device  = {device:?}");
    let (mut enc, ms_load) = ZunaEncoder::load(&config_path, &weights_path, device)?;
    eprintln!(
        "[encode_dir] encoder loaded in {ms_load:.0} ms — {}",
        enc.describe()
    );

    let data_cfg = DataConfig::default();
    let mut fifs: Vec<PathBuf> = std::fs::read_dir(&args.in_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("fif"))
        .collect();
    fifs.sort();
    if fifs.is_empty() {
        anyhow::bail!("no .fif in {}", args.in_dir.display());
    }
    eprintln!(
        "[encode_dir] {} files in {}",
        fifs.len(),
        args.in_dir.display()
    );

    let t_all = Instant::now();
    let mut total_epochs = 0usize;
    for (i, fif) in fifs.iter().enumerate() {
        let stem = fif.file_stem().and_then(|s| s.to_str()).unwrap_or("subj");
        let out = args.out_dir.join(format!("{stem}.zuna.safetensors"));
        let (n_ep, n_tok, dim, ms_pp, ms_enc) =
            encode_one_fif(&mut enc, fif, &out, &data_cfg, args.data_norm)?;
        total_epochs += n_ep;
        let mb = std::fs::metadata(&out)
            .map(|m| m.len() as f64 / 1e6)
            .unwrap_or(0.0);
        eprintln!(
            "[{:>2}/{}] {:>16} → {:>30}  [{} ep × {} tok × {} d = {:.1} MB; pp {:.0}ms enc {:.1}s]",
            i + 1,
            fifs.len(),
            fif.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            out.file_name().and_then(|s| s.to_str()).unwrap_or(""),
            n_ep,
            n_tok,
            dim,
            mb,
            ms_pp,
            ms_enc / 1000.0,
        );
    }
    eprintln!(
        "[encode_dir] {} files, {} epochs total in {:.1}s",
        fifs.len(),
        total_epochs,
        t_all.elapsed().as_secs_f64()
    );
    Ok(())
}
