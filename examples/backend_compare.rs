//! Burn vs RLX encoder benchmark across every compiled-in backend.
//!
//! Sweeps channel counts and times one encoder forward pass per
//! (engine, backend, channels). Skips backends that are not compiled in
//! or not available on this host.
//!
//! ```sh
//! # CPU-only quick compare:
//! cargo run --example backend_compare --release \
//!     --no-default-features --features burn,rlx,ndarray,rlx-cpu
//!
//! # Apple Silicon — Burn (wgpu/mlx) + RLX (metal/mlx):
//! cargo run --example backend_compare --release \
//!     --no-default-features \
//!     --features burn,rlx,ndarray,blas-accelerate,wgpu,mlx,rlx-cpu,rlx-metal,rlx-mlx
//! ```

#[path = "common/mod.rs"]
mod common;

use std::path::Path;
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "ZUNA — Burn vs RLX encoder benchmark (all compiled backends)")]
struct Args {
    #[arg(long, default_value = "4,8,12,19,32")]
    channels: String,

    #[arg(long, default_value_t = 3)]
    runs: usize,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, env = "RAYON_NUM_THREADS")]
    threads: Option<usize>,

    #[arg(long, default_value = common::DEFAULT_REPO, env = "ZUNA_REPO")]
    repo: String,

    #[arg(long, env = "HF_HOME")]
    hf_cache: Option<std::path::PathBuf>,

    #[arg(long, env = "ZUNA_WEIGHTS")]
    weights: Option<String>,

    #[arg(long, env = "ZUNA_CONFIG")]
    config: Option<String>,
}

struct BenchResult {
    engine: String,
    backend: String,
    n_channels: usize,
    n_tokens: usize,
    runs: Vec<f64>,
}

impl BenchResult {
    fn label(&self) -> String {
        format!("{}/{}", self.engine, self.backend)
    }
    fn min_ms(&self) -> f64 {
        self.runs.iter().cloned().fold(f64::INFINITY, f64::min)
    }
    fn mean_ms(&self) -> f64 {
        self.runs.iter().sum::<f64>() / self.runs.len() as f64
    }
}

// ── Synthetic RLX epoch ──────────────────────────────────────────────────────

fn make_synthetic_rlx_epoch(n_ch: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>, usize, usize) {
    let tf = zuna_rs::DataConfig::default().num_fine_time_pts;
    let tc = 40usize;
    let s = n_ch * tc;
    let mut eeg_tokens = vec![0f32; s * tf];
    for (i, v) in eeg_tokens.iter_mut().enumerate() {
        *v = (i as f32) * 0.001 - 0.5;
    }
    let mut tok_idx = vec![0i32; s * 4];
    for ch in 0..n_ch {
        let x_bin = (ch * 49 / n_ch.max(1)) as i32;
        let y_bin = ((ch * 7) % 50) as i32;
        for t in 0..tc {
            let row = ch * tc + t;
            tok_idx[row * 4] = x_bin;
            tok_idx[row * 4 + 1] = y_bin;
            tok_idx[row * 4 + 2] = 25;
            tok_idx[row * 4 + 3] = t as i32;
        }
    }
    let chan_pos = vec![0.01f32; n_ch * 3];
    (eeg_tokens, tok_idx, chan_pos, n_ch, tc)
}

// ── Burn backends ────────────────────────────────────────────────────────────

#[cfg(feature = "burn")]
mod burn_bench {
    use super::*;
    use burn::prelude::*;
    use burn::tensor::Distribution;
    use zuna_rs::data::InputBatch;
    use zuna_rs::ZunaEncoder;

    fn make_batch<B: Backend>(n_ch: usize, device: &B::Device) -> InputBatch<B> {
        let tf = zuna_rs::DataConfig::default().num_fine_time_pts;
        let tc = 40usize;
        let n_tokens = n_ch * tc;
        let encoder_input =
            Tensor::<B, 3>::random([1, n_tokens, tf], Distribution::Normal(0.0, 1.0), device);
        let mut idx_data = vec![0i64; n_tokens * 4];
        for ch in 0..n_ch {
            let x_bin = (ch * 49 / n_ch.max(1)) as i64;
            let y_bin = ((ch * 7) % 50) as i64;
            for t in 0..tc {
                let row = ch * tc + t;
                idx_data[row * 4] = x_bin;
                idx_data[row * 4 + 1] = y_bin;
                idx_data[row * 4 + 2] = 25;
                idx_data[row * 4 + 3] = t as i64;
            }
        }
        let tok_idx =
            Tensor::<B, 2, Int>::from_data(TensorData::new(idx_data, [n_tokens, 4]), device);
        let chan_pos = Tensor::<B, 2>::random([n_ch, 3], Distribution::Normal(0.0, 0.05), device);
        InputBatch {
            encoder_input,
            tok_idx,
            chan_pos,
            n_channels: n_ch,
            tc,
        }
    }

    fn bench_encoder<B: Backend>(
        engine: &str,
        backend: &str,
        enc: &ZunaEncoder<B>,
        channel_counts: &[usize],
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let device = enc.device();
        let mut out = Vec::new();
        for &n_ch in channel_counts {
            let n_tokens = n_ch * 40;
            for _ in 0..n_warmup {
                let b = make_batch::<B>(n_ch, device);
                let _ = enc.encode_batches(vec![b]);
            }
            let mut runs = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                let b = make_batch::<B>(n_ch, device);
                let t = Instant::now();
                let _ = enc.encode_batches(vec![b]);
                runs.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            out.push(BenchResult {
                engine: engine.into(),
                backend: backend.into(),
                n_channels: n_ch,
                n_tokens,
                runs,
            });
        }
        out
    }

    pub fn run_all(
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let mut all = Vec::new();

        #[cfg(feature = "ndarray")]
        {
            use burn::backend::{ndarray::NdArrayDevice, NdArray};
            let name = if cfg!(feature = "blas-accelerate") {
                "NdArray+Accelerate"
            } else if cfg!(feature = "openblas-system") {
                "NdArray+OpenBLAS"
            } else {
                "NdArray"
            };
            eprint!("  Burn/{name:<20} ");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ZunaEncoder::<NdArray>::load(config_path, weights_path, NdArrayDevice::Cpu)
            })) {
                Ok(Ok((enc, _))) => {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bench_encoder("Burn", name, &enc, channel_counts, n_warmup, n_runs)
                    })) {
                        Ok(r) => {
                            all.extend(r);
                            eprintln!("ok");
                        }
                        Err(_) => eprintln!("SKIP (panic during run)"),
                    }
                }
                Ok(Err(_)) => eprintln!("SKIP"),
                Err(_) => eprintln!("SKIP (panic on load)"),
            }
        }

        #[cfg(feature = "wgpu")]
        {
            use burn::backend::{wgpu::WgpuDevice, Wgpu};
            eprint!("  Burn/{:<20} ", "wgpu f32");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ZunaEncoder::<Wgpu>::load(config_path, weights_path, WgpuDevice::DefaultDevice)
            })) {
                Ok(Ok((enc, _))) => {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bench_encoder("Burn", "wgpu f32", &enc, channel_counts, n_warmup, n_runs)
                    })) {
                        Ok(r) => {
                            all.extend(r);
                            eprintln!("ok");
                        }
                        Err(_) => eprintln!("SKIP (panic during run)"),
                    }
                }
                Ok(Err(_)) => eprintln!("SKIP"),
                Err(_) => eprintln!("SKIP (panic on load)"),
            }
        }

        #[cfg(feature = "mlx")]
        {
            use burn_mlx::{Mlx, MlxDevice};
            eprint!("  Burn/{:<20} ", "MLX f32");
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ZunaEncoder::<Mlx>::load(config_path, weights_path, MlxDevice::Gpu)
            })) {
                Ok(Ok((enc, _))) => {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bench_encoder("Burn", "MLX f32", &enc, channel_counts, n_warmup, n_runs)
                    })) {
                        Ok(r) => {
                            all.extend(r);
                            eprintln!("ok");
                        }
                        Err(_) => eprintln!("SKIP (panic during run)"),
                    }
                }
                Ok(Err(_)) => eprintln!("SKIP"),
                Err(_) => eprintln!("SKIP (panic on load)"),
            }
        }

        all
    }
}

// ── RLX backends ─────────────────────────────────────────────────────────────

#[cfg(feature = "rlx")]
mod rlx_bench {
    use super::*;
    use zuna_rs::rlx::ZunaEncoder;

    fn bench_encoder(
        backend: &str,
        enc: &mut ZunaEncoder,
        channel_counts: &[usize],
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let mut out = Vec::new();
        for &n_ch in channel_counts {
            let (eeg, tok, pos, nc, tc) = make_synthetic_rlx_epoch(n_ch);
            let n_tokens = n_ch * tc;
            for _ in 0..n_warmup {
                let (e, t, p, c, tc_) = make_synthetic_rlx_epoch(n_ch);
                let _ = enc.encode_one(&e, &t, &p, c, tc_);
            }
            let mut runs = Vec::with_capacity(n_runs);
            for _ in 0..n_runs {
                let (e, t, p, c, tc_) = make_synthetic_rlx_epoch(n_ch);
                let t0 = Instant::now();
                let _ = enc.encode_one(&e, &t, &p, c, tc_).expect("encode_one");
                runs.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let _ = (eeg, tok, pos, nc, tc);
            out.push(BenchResult {
                engine: "RLX".into(),
                backend: backend.into(),
                n_channels: n_ch,
                n_tokens,
                runs,
            });
        }
        out
    }

    fn try_bench(
        label: &str,
        device: rlx::Device,
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        eprint!("  RLX/{label:<20} ");
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ZunaEncoder::load(config_path, weights_path, device)
        })) {
            Ok(Ok((mut enc, _))) => {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    bench_encoder(label, &mut enc, channel_counts, n_warmup, n_runs)
                }));
                match r {
                    Ok(v) => {
                        eprintln!("ok");
                        v
                    }
                    Err(_) => {
                        eprintln!("SKIP (panic during run)");
                        Vec::new()
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("SKIP ({e})");
                Vec::new()
            }
            Err(_) => {
                eprintln!("SKIP (panic on load)");
                Vec::new()
            }
        }
    }

    pub fn run_all(
        config_path: &Path,
        weights_path: &Path,
        channel_counts: &[usize],
        n_warmup: usize,
        n_runs: usize,
    ) -> Vec<BenchResult> {
        let mut all = Vec::new();

        #[cfg(feature = "rlx-cpu")]
        {
            let name = if cfg!(feature = "rlx-blas-accelerate") {
                "CPU+Accelerate"
            } else {
                "CPU"
            };
            all.extend(try_bench(
                name,
                rlx::Device::Cpu,
                config_path,
                weights_path,
                channel_counts,
                n_warmup,
                n_runs,
            ));
        }

        #[cfg(feature = "rlx-metal")]
        all.extend(try_bench(
            "Metal",
            rlx::Device::Metal,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        #[cfg(feature = "rlx-mlx")]
        all.extend(try_bench(
            "MLX",
            rlx::Device::Mlx,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        // TODO: ZUNA encoder on RLX wgpu once rlx-wgpu lowers FusedResidualRmsNorm
        // (currently panics — see rlx-wgpu backend.rs). Re-enable in backend_compare
        // and treat as a supported backend when that lands upstream.
        #[cfg(feature = "rlx-gpu")]
        all.extend(try_bench(
            "wgpu",
            rlx::Device::Gpu,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        #[cfg(feature = "rlx-cuda")]
        all.extend(try_bench(
            "CUDA",
            rlx::Device::Cuda,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        #[cfg(feature = "rlx-rocm")]
        all.extend(try_bench(
            "ROCm",
            rlx::Device::Rocm,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        #[cfg(feature = "rlx-tpu")]
        all.extend(try_bench(
            "TPU",
            rlx::Device::Tpu,
            config_path,
            weights_path,
            channel_counts,
            n_warmup,
            n_runs,
        ));

        all
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _n = zuna_rs::init_threads(args.threads);

    let channel_counts: Vec<usize> = args
        .channels
        .split(',')
        .map(|s| s.trim().parse().expect("invalid channel count"))
        .collect();

    let (weights_path, config_path) = common::resolve_weights(
        &args.repo,
        args.weights.as_deref(),
        args.config.as_deref(),
        args.hf_cache.as_deref(),
    )?;

    println!("=== ZUNA Burn vs RLX — backend benchmark ===");
    println!("  weights  : {}", weights_path.display());
    println!("  channels : {:?}", channel_counts);
    println!("  runs     : {} (+ {} warmup)", args.runs, args.warmup);
    println!();

    let mut results = Vec::new();

    #[cfg(feature = "burn")]
    {
        println!("Burn backends:");
        results.extend(burn_bench::run_all(
            &config_path,
            &weights_path,
            &channel_counts,
            args.warmup,
            args.runs,
        ));
        println!();
    }

    #[cfg(feature = "rlx")]
    {
        println!("RLX backends:");
        results.extend(rlx_bench::run_all(
            &config_path,
            &weights_path,
            &channel_counts,
            args.warmup,
            args.runs,
        ));
        println!();
    }

    if results.is_empty() {
        anyhow::bail!(
            "no backends ran — enable `rlx` (+ an rlx-* backend) and/or `burn` (+ `ndarray`, …)"
        );
    }

    println!(
        "{:<28} {:>4} {:>6} {:>10} {:>10}",
        "Engine/Backend", "Ch", "Tok", "Min(ms)", "Mean(ms)"
    );
    println!("{}", "─".repeat(62));
    for r in &results {
        println!(
            "{:<28} {:>4} {:>6} {:>10.1} {:>10.1}",
            r.label(),
            r.n_channels,
            r.n_tokens,
            r.min_ms(),
            r.mean_ms(),
        );
    }

    // Speedup: RLX vs Burn per channel (best backend each)
    #[cfg(all(feature = "burn", feature = "rlx"))]
    {
        println!();
        println!("── RLX best / Burn best (speedup > 1 means RLX faster) ──");
        for &ch in &channel_counts {
            let burn_best = results
                .iter()
                .filter(|r| r.engine == "Burn" && r.n_channels == ch)
                .map(|r| r.min_ms())
                .fold(f64::INFINITY, f64::min);
            let rlx_best = results
                .iter()
                .filter(|r| r.engine == "RLX" && r.n_channels == ch)
                .map(|r| r.min_ms())
                .fold(f64::INFINITY, f64::min);
            if burn_best.is_finite() && rlx_best.is_finite() {
                println!(
                    "  {ch:>3} ch: {:.2}x  (Burn {:.0} ms vs RLX {:.0} ms)",
                    burn_best / rlx_best,
                    burn_best,
                    rlx_best
                );
            }
        }
    }

    println!();
    println!("── CSV ──");
    println!("engine,backend,channels,tokens,min_ms,mean_ms");
    for r in &results {
        println!(
            "{},{},{},{},{:.1},{:.1}",
            r.engine,
            r.backend,
            r.n_channels,
            r.n_tokens,
            r.min_ms(),
            r.mean_ms(),
        );
    }

    Ok(())
}
