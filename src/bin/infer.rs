//! ZUNA EEG inference — thin CLI over [`zuna_rs::ZunaInference`].
//!
//! Build:
//!   cargo build --release                              # CPU (default)
//!   cargo build --release --features blas-accelerate   # macOS Accelerate
//!   cargo build --release --features metal             # Apple Metal native
//!   cargo build --release --features mlx               # Apple MLX
//!
//! Usage:
//! ```text
//!   infer --weights <st> --config <json> --fif <fif> --output <st>
//!         [--device cpu|metal|mlx|gpu|cuda]
//!         [--steps 50] [--cfg 1.0] [--data-norm 10.0] [--verbose]
//! ```

use std::path::Path;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use zuna_rs::rlx::{EpochOutput, ZunaInference};
use zuna_rs::{preprocess_fif_cpu, DataConfig};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
    Mlx,
    Gpu,
    Cuda,
    Rocm,
    Tpu,
}

impl DeviceArg {
    fn into_rlx(self) -> rlx::Device {
        match self {
            Self::Cpu => rlx::Device::Cpu,
            Self::Metal => rlx::Device::Metal,
            Self::Mlx => rlx::Device::Mlx,
            Self::Gpu => rlx::Device::Gpu,
            Self::Cuda => rlx::Device::Cuda,
            Self::Rocm => rlx::Device::Rocm,
            Self::Tpu => rlx::Device::Tpu,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "ZUNA EEG model inference (RLX runtime)")]
struct Args {
    /// Compute device.
    #[arg(long, default_value = "cpu")]
    device: DeviceArg,

    /// Safetensors weights file (HuggingFace Zyphra/ZUNA or Zyphra/ZUNA1.1).
    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: String,

    /// config.json matching --weights (Zyphra/ZUNA or Zyphra/ZUNA1.1).
    #[arg(long, env = "ZUNA_CONFIG")]
    config: String,

    /// Raw EEG recording (.fif).
    #[arg(long)]
    fif: String,

    /// Output safetensors file.
    #[arg(long)]
    output: String,

    /// Diffusion denoising steps (50 = full quality, 10 = fast preview).
    #[arg(long, default_value_t = 50)]
    steps: usize,

    /// Classifier-free guidance scale (1.0 = off).
    #[arg(long, default_value_t = 1.0)]
    cfg: f32,

    /// Signal normalisation divisor (applied before model, inverted after).
    #[arg(long, default_value_t = 10.0)]
    data_norm: f32,

    /// Seed for the rectified-flow initial noise. Same seed → same output.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Number of CPU threads for the CPU backend (0 or omit = all cores).
    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    /// Print model config, electrode positions, per-epoch stats.
    #[arg(long, short = 'v')]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let n_threads = zuna_rs::init_threads(args.threads);
    let device = args.device.into_rlx();
    let t0 = Instant::now();

    eprintln!("Device   : {:?}  ({n_threads} threads)", device);

    // ── Load model ─────────────────────────────────────────────────────────
    let (mut zuna, ms_load) =
        ZunaInference::load(Path::new(&args.config), Path::new(&args.weights), device)?;
    let arch = zuna.arch();
    let cfg = zuna.model_cfg();
    eprintln!(
        "Model    : {} dim={} layers={} head_dim={} t_dim={}  ({ms_load:.0} ms)",
        arch.label(),
        cfg.dim,
        cfg.n_layers,
        cfg.head_dim,
        cfg.t_dim,
    );

    // ── Preprocess ─────────────────────────────────────────────────────────
    let t_pp = Instant::now();
    let pre = preprocess_fif_cpu(Path::new(&args.fif), &DataConfig::default(), args.data_norm)?;
    let ms_preproc = t_pp.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Input    : {}  →  {} epochs ({} channels, {} tokens/epoch)",
        args.fif,
        pre.epochs.len(),
        pre.info.ch_names.len(),
        pre.epochs.first().map(|e| e.s).unwrap_or(0),
    );
    eprintln!("Preproc  : {ms_preproc:.1} ms");

    if args.verbose {
        eprintln!("── Electrode positions (MNI head frame, mm) ──────");
        eprintln!(
            "  {:<4} {:<8} {:>10} {:>10} {:>10}",
            "#", "Name", "Right(x)", "Ant(y)", "Sup(z)"
        );
        for (i, (name, pos)) in pre
            .info
            .ch_names
            .iter()
            .zip(&pre.info.ch_pos_mm)
            .enumerate()
        {
            eprintln!(
                "  {:<4} {:<8} {:>10.2} {:>10.2} {:>10.2}",
                i, name, pos[0], pos[1], pos[2]
            );
        }
    }

    // ── Run pipeline ───────────────────────────────────────────────────────
    let t_inf = Instant::now();
    let mut outputs = Vec::with_capacity(pre.epochs.len());
    for (i, ep) in pre.epochs.iter().enumerate() {
        let seed = args.seed.wrapping_add(i as u64).max(1);
        let out = zuna.run_epoch(ep, args.steps, args.cfg, args.data_norm, seed)?;
        outputs.push(out);
    }
    let ms_infer = t_inf.elapsed().as_secs_f64() * 1000.0;
    let n = outputs.len();
    eprintln!(
        "Infer    : {ms_infer:.0} ms  ({n} × {} steps; {:.0} ms/epoch)",
        args.steps,
        if n > 0 { ms_infer / n as f64 } else { 0.0 },
    );

    if args.verbose {
        for (i, ep) in outputs.iter().enumerate() {
            let data = &ep.reconstructed;
            let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
            let var: f64 = data
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / data.len() as f64;
            let std = var.sqrt();
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "  [ep {}/{n}] {:?}  mean={mean:.4}  std={std:.4}  min={min:.4}  max={max:.4}",
                i + 1,
                ep.shape,
            );
        }
    }

    // ── Save ───────────────────────────────────────────────────────────────
    save_safetensors(&outputs, &args.output)?;
    eprintln!(
        "Output   → {}  ({:.0} ms total)",
        args.output,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    eprintln!(
        "TIMING weights={ms_load:.1}ms preproc={ms_preproc:.1}ms inference={ms_infer:.1}ms total={:.1}ms",
        t0.elapsed().as_secs_f64() * 1000.0,
    );
    Ok(())
}

/// Write reconstructed epochs to a safetensors file.
///
/// Keys per epoch `N`:
/// * `reconstructed_N` — `[C, T]` float32
/// * `chan_pos_N`       — `[C, 3]` float32
///
/// Plus a scalar `n_samples` float32.
fn save_safetensors(epochs: &[EpochOutput], path: &str) -> anyhow::Result<()> {
    use safetensors::{Dtype, View};
    use std::borrow::Cow;

    struct F32Tensor {
        data: Vec<u8>,
        shape: Vec<usize>,
    }
    impl View for F32Tensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
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
    fn to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    let mut keys: Vec<String> = Vec::new();
    let mut tensors: Vec<F32Tensor> = Vec::new();
    for (i, ep) in epochs.iter().enumerate() {
        keys.push(format!("reconstructed_{i}"));
        tensors.push(F32Tensor {
            data: to_bytes(&ep.reconstructed),
            shape: ep.shape.clone(),
        });
        keys.push(format!("chan_pos_{i}"));
        tensors.push(F32Tensor {
            data: to_bytes(&ep.chan_pos),
            shape: vec![ep.n_channels, 3],
        });
    }
    keys.push("n_samples".into());
    tensors.push(F32Tensor {
        data: to_bytes(&[epochs.len() as f32]),
        shape: vec![1],
    });

    let pairs: Vec<(&str, F32Tensor)> = keys.iter().map(|s| s.as_str()).zip(tensors).collect();
    let bytes = safetensors::serialize(pairs, None)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
