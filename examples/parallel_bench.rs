/// Quick benchmark for multi-file parallel encoding.
///
/// Usage:
///   cargo run --example parallel_bench --release --features blas-accelerate

#[path = "common/mod.rs"]
mod common;

use std::path::PathBuf;
use std::time::Instant;
use zuna_rs::ZunaEncoder;

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "ndarray"))]
    { anyhow::bail!("parallel_bench requires --features ndarray"); }

    #[cfg(feature = "ndarray")]
    run()
}

#[cfg(feature = "ndarray")]
fn run() -> anyhow::Result<()> {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    let n_threads = zuna_rs::init_threads(None);
    println!("Threads: {n_threads}");

    let (weights_path, config_path) = common::resolve_weights(
        "Zyphra/ZUNA", None, None, None,
    )?;

    let device = NdArrayDevice::Cpu;
    let (enc, ms_load) = ZunaEncoder::<NdArray>::load(&config_path, &weights_path, device)?;
    println!("Model loaded in {ms_load:.0} ms\n");

    let fif = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif"));

    // --- Section 1: parallel preprocessing benefit ---
    println!("=== Multi-file parallel preprocessing + batched encode ===");
    for n_files in [1, 3, 5, 10, 20] {
        let paths: Vec<PathBuf> = (0..n_files).map(|_| fif.clone()).collect();

        // Sequential
        let t = Instant::now();
        for p in &paths {
            let _ = enc.encode_fif(p, 10.0)?;
        }
        let ms_seq = t.elapsed().as_secs_f64() * 1000.0;

        // Parallel
        let t = Instant::now();
        let results = enc.encode_fif_parallel(&paths, 10.0)?;
        let ms_par = t.elapsed().as_secs_f64() * 1000.0;

        let total_epochs: usize = results.iter().map(|r| r.epochs.len()).sum();
        let speedup = ms_seq / ms_par;
        let ms_per_epoch = ms_par / total_epochs as f64;
        println!(
            "{n_files:>2} files ({total_epochs:>3} epochs):  seq={ms_seq:>8.0}ms  par={ms_par:>8.0}ms  \
             speedup={speedup:.2}x  {ms_per_epoch:.0}ms/epoch"
        );
    }

    // --- Section 2: per-epoch cost at different batch sizes ---
    println!("\n=== Batch size scaling (single-file, 3 epochs each) ===");
    // Preprocess once, then replicate epochs to simulate larger batches
    let (batches, _) = enc.preprocess_fif(&fif, 10.0)?;
    println!("Baseline: {} epochs from sample FIF", batches.len());

    for n_copies in [1, 2, 5, 10, 20, 50] {
        let mut all_batches = Vec::new();
        for _ in 0..n_copies {
            for b in &batches {
                all_batches.push(zuna_rs::InputBatch {
                    encoder_input: b.encoder_input.clone(),
                    tok_idx: b.tok_idx.clone(),
                    chan_pos: b.chan_pos.clone(),
                    n_channels: b.n_channels,
                    tc: b.tc,
                });
            }
        }
        let total = all_batches.len();

        let t = Instant::now();
        let _ = enc.encode_batches(all_batches)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let ms_per = ms / total as f64;
        println!("  batch={total:>4} epochs:  {ms:>8.0}ms total  {ms_per:>6.0}ms/epoch");
    }

    Ok(())
}
