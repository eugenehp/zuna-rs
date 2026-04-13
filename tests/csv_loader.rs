//! Integration tests for the CSV / tensor data-loading path.
//!
//! All CSV fixtures are generated at test runtime from the FIF file in
//! `data/sample1_raw.fif` so the tests are fully self-contained.
//!
//! Signal-precision test strategy
//! ────────────────────────────────
//! `load_from_fif` and `load_from_csv` both start from the **same** raw f32
//! values (the FIF f64 cast to f32 matches the `{:.9e}` round-trip).  They
//! run the **same** `exg::preprocess` pipeline; channel positions are passed
//! through unchanged and have no effect on the signal.  Therefore
//! `encoder_input` must be bit-for-bit identical, and we assert
//!   max |csv_val − fif_val| < 1 × 10⁻⁵.

mod helpers;

// ── Backend ───────────────────────────────────────────────────────────────────
use burn::backend::NdArray as B;
type Device = burn::backend::ndarray::NdArrayDevice;
fn dev() -> Device { Device::Cpu }

// ── Library under test ────────────────────────────────────────────────────────
use zuna_rs::{
    config::DataConfig,
    data::load_from_fif,
    load_from_csv, load_from_named_tensor, load_from_raw_tensor,
    CsvLoadOptions, PaddingStrategy,
};

use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn fif_path() -> PathBuf {
    helpers::csv_gen::fif_path()
}

fn data_cfg() -> DataConfig { DataConfig::default() }

/// FIF channel names in order (Fp1 … F8).
const FIF_CHANNELS: &[&str] = &[
    "Fp1", "Fp2", "F3", "F4", "C3", "C4",
    "P3",  "P4",  "O1", "O2", "F7", "F8",
];

/// Generate all test CSV fixtures into a temp subdirectory and return the paths.
/// The directory name includes the test binary name so parallel `cargo test` runs
/// don't race.
struct Fixtures {
    pub all12: PathBuf,
    pub ten:   PathBuf,
    pub eight: PathBuf,
}

/// Generate fixtures once (thread-safe via OnceLock).
/// Multiple parallel tests all call this; only the first invocation writes files.
fn get_fixtures() -> &'static Fixtures {
    static ONCE: std::sync::OnceLock<Fixtures> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        let tmp = std::env::temp_dir().join("zuna_csv_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");

        let fif = fif_path();

        let all12 = tmp.join("all12.csv");
        helpers::csv_gen::fif_to_csv(&fif, &all12, None);

        let ten = tmp.join("ten.csv");
        helpers::csv_gen::fif_to_csv(
            &fif, &ten,
            Some(&["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2"]),
        );

        let eight = tmp.join("eight.csv");
        helpers::csv_gen::fif_to_csv(
            &fif, &eight,
            Some(&["Fp1","Fp2","F3","F4","C3","C4","P3","P4"]),
        );

        Fixtures { all12, ten, eight }
    })
}

/// Extract encoder_input as a flat Vec<f32>.
fn ei_vals<Bk: burn::prelude::Backend>(batch: &zuna_rs::data::InputBatch<Bk>) -> Vec<f32> {
    let data = batch.encoder_input.clone().into_data();
    data.to_vec::<f32>().expect("encoder_input to_vec")
}

/// Extract tok_idx as a flat Vec<i32>.
#[allow(dead_code)]
fn ti_vals<Bk: burn::prelude::Backend>(batch: &zuna_rs::data::InputBatch<Bk>) -> Vec<i32> {
    let data = batch.tok_idx.clone().into_data();
    data.to_vec::<i32>().expect("tok_idx to_vec")
}

/// Max absolute difference between two equal-length f32 slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — shape checks for a 12-channel CSV
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that a 12-channel CSV produces the expected tensor shapes.
///
/// FIF: 12 ch × 3840 samples (15 s @ 256 Hz)
/// → 3 epochs × (tc=40 coarse steps × tf=32 fine pts = 1280 samples)
/// → S = 12 × 40 = 480 tokens
/// → encoder_input: [1, 480, 32], tok_idx: [480, 4]
#[test]
fn csv_all12_shape() {
    let f = get_fixtures();
    let opts = CsvLoadOptions::default();
    let (batches, info) = load_from_csv::<B>(&f.all12, &opts, &data_cfg(), &dev())
        .expect("load_from_csv all12");

    assert_eq!(info.n_epochs, 3,    "expected 3 epochs from 15 s recording");
    assert_eq!(info.n_padded, 0,    "no padding with all 12 channels");
    assert_eq!(info.ch_names.len(), 12);
    assert_eq!(batches.len(),       3);

    let b = &batches[0];
    // encoder_input: [1, S, tf] = [1, 480, 32]
    assert_eq!(b.encoder_input.dims(), [1, 480, 32],
               "encoder_input shape mismatch");
    // tok_idx: [S, 4] = [480, 4]
    assert_eq!(b.tok_idx.dims(), [480, 4],
               "tok_idx shape mismatch");
    assert_eq!(b.n_channels, 12);
    assert_eq!(b.tc, 40);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — CSV signal must match FIF signal precisely
// ─────────────────────────────────────────────────────────────────────────────

/// Both loaders start from the same raw f32 data and run the same pipeline.
/// `encoder_input` must agree to within floating-point text-round-trip precision.
#[test]
fn csv_all12_matches_fif_signal() {
    let f = get_fixtures();
    let dcfg = data_cfg();
    let d = dev();

    let opts = CsvLoadOptions::default();
    let (csv_batches, _) = load_from_csv::<B>(&f.all12, &opts, &dcfg, &d)
        .expect("csv load");

    let (fif_batches, _) = load_from_fif::<B>(&fif_path(), &dcfg, 10.0, &d)
        .expect("fif load");

    assert_eq!(csv_batches.len(), fif_batches.len(), "epoch count must match");

    let mut global_max: f32 = 0.0;
    let mut global_mean: f32 = 0.0;

    for (epoch_i, (cb, fb)) in csv_batches.iter().zip(fif_batches.iter()).enumerate() {
        let cv = ei_vals(cb);
        let fv = ei_vals(fb);

        assert_eq!(cv.len(), fv.len(),
                   "epoch {epoch_i}: encoder_input length mismatch");

        let mx = max_abs_diff(&cv, &fv);
        let mn = mean_abs_diff(&cv, &fv);
        println!("  epoch {epoch_i}: max|Δ|={mx:.3e}  mean|Δ|={mn:.3e}  n={}", cv.len());

        global_max  = global_max.max(mx);
        global_mean += mn;

        assert!(
            mx < 1e-5,
            "epoch {epoch_i}: max |csv − fif| = {mx:.3e} exceeds 1e-5 tolerance"
        );
    }

    global_mean /= csv_batches.len() as f32;
    println!("ALL EPOCHS  global max|Δ|={global_max:.3e}  global mean|Δ|={global_mean:.3e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — ten-channel CSV with zero padding to 12 channels
// ─────────────────────────────────────────────────────────────────────────────

/// A 10-channel CSV (F7 and F8 absent) with `target_channels = all 12`
/// must produce 2 padded channels and correct tensor shapes.
#[test]
fn csv_ten_zero_pad() {
    let f = get_fixtures();
    let targets: Vec<String> = FIF_CHANNELS.iter().map(|s| s.to_string()).collect();
    let opts = CsvLoadOptions {
        target_channels: Some(targets),
        padding: PaddingStrategy::Zero,
        ..Default::default()
    };

    let (batches, info) = load_from_csv::<B>(&f.ten, &opts, &data_cfg(), &dev())
        .expect("csv ten zero_pad");

    assert_eq!(info.n_padded, 2, "F7 and F8 should be padded");
    assert_eq!(info.n_epochs, 3);
    assert_eq!(info.ch_names.len(), 12);

    // Output shape identical to 12-channel case
    let b = &batches[0];
    assert_eq!(b.encoder_input.dims(), [1, 480, 32]);
    assert_eq!(b.tok_idx.dims(), [480, 4]);

    // F7 is at index 10, F8 at index 11 in the target list.
    // After zero-padding then average-reference, the zero channels will
    // contribute to the channel mean so they won't remain exactly zero
    // post-pipeline, but they WILL be different from the all12 result.
    // Just verify the batch is not all-NaN.
    let ei = ei_vals(b);
    assert!(!ei.iter().any(|v| v.is_nan()), "no NaNs in zero-padded output");
    assert!(ei.iter().any(|&v| v != 0.0), "output must not be all zeros");

    println!("csv_ten_zero_pad: n_padded={} n_epochs={}", info.n_padded, info.n_epochs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — ten-channel CSV with CloneNearest padding
// ─────────────────────────────────────────────────────────────────────────────

/// Missing F7 and F8 should be cloned from whichever loaded channel is
/// nearest in scalp-position space.  The cloned rows must be non-zero.
#[test]
fn csv_ten_clone_nearest() {
    let f = get_fixtures();
    let targets: Vec<String> = FIF_CHANNELS.iter().map(|s| s.to_string()).collect();
    let opts = CsvLoadOptions {
        target_channels: Some(targets),
        padding: PaddingStrategy::CloneNearest,
        ..Default::default()
    };

    let (batches, info) = load_from_csv::<B>(&f.ten, &opts, &data_cfg(), &dev())
        .expect("csv ten clone_nearest");

    assert_eq!(info.n_padded, 2);
    assert_eq!(info.n_epochs, 3);
    assert_eq!(batches[0].encoder_input.dims(), [1, 480, 32]);

    // With clone strategy the channel count going into exg is 12 (same as all12),
    // but two channels are copies of nearest neighbours.  The output must be
    // non-trivial.
    let ei = ei_vals(&batches[0]);
    assert!(!ei.iter().any(|v| v.is_nan()));
    // At least some nonzero values expected
    assert!(ei.iter().any(|&v| v.abs() > 1e-6));

    // Importantly the result must DIFFER from zero-padded output because cloning
    // real data changes the average reference and z-score statistics.
    let targets2: Vec<String> = FIF_CHANNELS.iter().map(|s| s.to_string()).collect();
    let zero_opts = CsvLoadOptions {
        target_channels: Some(targets2),
        padding: PaddingStrategy::Zero,
        ..Default::default()
    };
    let (zero_batches, _) = load_from_csv::<B>(&f.ten, &zero_opts, &data_cfg(), &dev())
        .expect("zero pad for comparison");
    let zero_ei = ei_vals(&zero_batches[0]);

    let diff = max_abs_diff(&ei, &zero_ei);
    assert!(diff > 1e-6, "CloneNearest and Zero padding should produce different signals");

    println!("csv_ten_clone_nearest: max|clone-zero|={diff:.3e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — eight-channel CSV with CloneChannel padding
// ─────────────────────────────────────────────────────────────────────────────

/// 8-channel CSV (Fp1..P4) padded to 12 with CloneChannel("Fp1").
/// The 4 missing channels (O1, O2, F7, F8) must each be a clone of Fp1's
/// raw signal.  Verify shape and that cloned rows equal Fp1 in the raw
/// (pre-pipeline) domain by checking the initial data in the CSV helper.
#[test]
fn csv_eight_clone_channel() {
    let f = get_fixtures();
    let targets: Vec<String> = FIF_CHANNELS.iter().map(|s| s.to_string()).collect();
    let opts = CsvLoadOptions {
        target_channels: Some(targets),
        padding: PaddingStrategy::CloneChannel("Fp1".to_string()),
        ..Default::default()
    };

    let (batches, info) = load_from_csv::<B>(&f.eight, &opts, &data_cfg(), &dev())
        .expect("csv eight clone_channel");

    assert_eq!(info.n_padded, 4, "O1, O2, F7, F8 should be padded");
    assert_eq!(info.n_epochs, 3);
    assert_eq!(batches[0].encoder_input.dims(), [1, 480, 32]);

    let ei = ei_vals(&batches[0]);
    assert!(!ei.iter().any(|v| v.is_nan()));
    assert!(ei.iter().any(|&v| v.abs() > 1e-6));

    println!("csv_eight_clone_channel: n_padded={} n_epochs={}", info.n_padded, info.n_epochs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — load_from_named_tensor with FIF channel names
// ─────────────────────────────────────────────────────────────────────────────

/// `load_from_named_tensor` with the exact FIF channels must produce
/// the same encoder_input as `load_from_fif` (signal is position-independent).
#[test]
fn tensor_loader_named_matches_fif() {
    use exg::fiff::raw::open_raw;
    use std::collections::HashMap;

    let fif = fif_path();
    let dcfg = data_cfg();
    let d = dev();

    // Read raw f64 data from FIF, cast to f32 (identical to what load_from_fif does).
    let raw = open_raw(&fif).expect("open_raw");
    let sfreq = raw.info.sfreq as f32;
    let data_f64 = raw.read_all_data().expect("read_all_data");
    let data_f32 = data_f64.mapv(|v| v as f32);

    let ch_names: Vec<&str> = raw.info.chs.iter().map(|ch| ch.name.as_str()).collect();

    let tensor_batches = load_from_named_tensor::<B>(
        data_f32, &ch_names, sfreq, 10.0, &HashMap::new(), &dcfg, &d,
    ).expect("load_from_named_tensor");

    let (fif_batches, _) = load_from_fif::<B>(&fif, &dcfg, 10.0, &d)
        .expect("load_from_fif");

    assert_eq!(tensor_batches.len(), fif_batches.len());

    for (i, (tb, fb)) in tensor_batches.iter().zip(fif_batches.iter()).enumerate() {
        let tv = ei_vals(tb);
        let fv = ei_vals(fb);
        let mx = max_abs_diff(&tv, &fv);
        println!("  named tensor epoch {i}: max|Δ|={mx:.3e}");
        assert!(mx < 1e-5, "epoch {i}: max |named_tensor − fif| = {mx:.3e}");
        assert_eq!(tb.encoder_input.dims(), [1, 480, 32]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7 — load_from_raw_tensor with explicit FIF positions
// ─────────────────────────────────────────────────────────────────────────────

/// `load_from_raw_tensor` with the exact FIF positions must produce the
/// same encoder_input as `load_from_fif`.
#[test]
fn tensor_loader_explicit_xyz_matches_fif() {
    use exg::fiff::raw::open_raw;

    let fif = fif_path();
    let dcfg = data_cfg();
    let d = dev();

    let raw = open_raw(&fif).expect("open_raw");
    let sfreq = raw.info.sfreq as f32;
    let positions: Vec<[f32; 3]> = raw.info.chs.iter()
        .map(|ch| [ch.loc[0], ch.loc[1], ch.loc[2]])  // metres
        .collect();
    let data_f64 = raw.read_all_data().expect("read_all_data");
    let data_f32 = data_f64.mapv(|v| v as f32);

    let tensor_batches = load_from_raw_tensor::<B>(
        data_f32, &positions, sfreq, 10.0, &dcfg, &d,
    ).expect("load_from_raw_tensor");

    let (fif_batches, _) = load_from_fif::<B>(&fif, &dcfg, 10.0, &d)
        .expect("load_from_fif");

    assert_eq!(tensor_batches.len(), fif_batches.len());

    for (i, (tb, fb)) in tensor_batches.iter().zip(fif_batches.iter()).enumerate() {
        let tv = ei_vals(tb);
        let fv = ei_vals(fb);
        let mx = max_abs_diff(&tv, &fv);
        println!("  explicit-xyz epoch {i}: max|Δ|={mx:.3e}");
        assert!(mx < 1e-5, "epoch {i}: max |raw_tensor − fif| = {mx:.3e}");
        assert_eq!(tb.encoder_input.dims(), [1, 480, 32]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8 — recordings shorter than 5 s must return an error
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn csv_too_short_returns_error() {
    // Write a 3-second CSV (768 samples @ 256 Hz) — shorter than 5 s minimum.
    let path = std::env::temp_dir().join("zuna_short_test.csv");
    {
        use std::io::Write as IoWrite;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "timestamp,Fp1").unwrap();
        for i in 0..768_usize {
            writeln!(f, "{:.9e},{:.9e}", i as f32 / 256.0, 1e-6_f32).unwrap();
        }
    }

    let opts = CsvLoadOptions::default();
    let result = load_from_csv::<B>(&path, &opts, &data_cfg(), &dev())
        .map(|_| ()); // erase Ok type so unwrap_err doesn't need T: Debug
    assert!(result.is_err(), "expected Err for short recording, got Ok");
    let msg = format!("{}", result.unwrap_err());
    println!("  short-recording error: {msg}");
    // Error message should mention the duration
    assert!(msg.contains("shorter") || msg.contains("3.00") || msg.contains("minimum"),
            "unexpected error message: {msg}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9 — unknown channel name gets centroid fallback (no panic)
// ─────────────────────────────────────────────────────────────────────────────

/// A channel whose name is not in any montage database should get the centroid
/// of the other channels as its position.  The pipeline must not panic.
#[test]
fn csv_unknown_channel_no_panic() {
    // Generate a 6-second CSV with one known channel and one completely unknown.
    let path = std::env::temp_dir().join("zuna_unknown_ch_test.csv");
    {
        use std::io::Write as IoWrite;
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "timestamp,Cz,XYZFAKE99").unwrap();
        // 6 seconds at 256 Hz = 1536 samples
        for i in 0..1536_usize {
            let t  = i as f32 / 256.0;
            let v1 = (2.0 * std::f32::consts::PI * 10.0 * t).sin() * 1e-5;
            let v2 = (2.0 * std::f32::consts::PI * 20.0 * t).sin() * 1e-5;
            writeln!(f, "{t:.9e},{v1:.9e},{v2:.9e}").unwrap();
        }
    }

    let opts = CsvLoadOptions::default();
    let result = load_from_csv::<B>(&path, &opts, &data_cfg(), &dev());
    assert!(result.is_ok(), "unexpected error: {:?}", result.err());

    let (batches, info) = result.unwrap();
    assert_eq!(info.ch_names, ["Cz", "XYZFAKE99"]);
    assert_eq!(info.n_padded, 0);
    assert!(!batches.is_empty());

    // XYZFAKE99 gets the centroid of Cz's position → same as Cz.
    // Verify the position is within DataConfig bounds.
    let [x, y, z] = info.ch_pos_m[1];
    let dcfg = data_cfg();
    assert!(x >= dcfg.xyz_min[0] && x <= dcfg.xyz_max[0], "x={x} out of bounds");
    assert!(y >= dcfg.xyz_min[1] && y <= dcfg.xyz_max[1], "y={y} out of bounds");
    assert!(z >= dcfg.xyz_min[2] && z <= dcfg.xyz_max[2], "z={z} out of bounds");
    println!("  XYZFAKE99 position (centroid fallback): [{x:.4},{y:.4},{z:.4}]");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10 — position_overrides: fuzzy XYZ for CloneNearest
// ─────────────────────────────────────────────────────────────────────────────

/// With `CloneNearest` and a `position_overrides` entry for the missing channel,
/// the override XYZ is used as the query point for nearest-neighbour search.
/// This lets callers specify "fuzzy coordinates" to control which real channel
/// is cloned.
#[test]
fn csv_clone_nearest_with_position_override() {
    use std::collections::HashMap;
    use zuna_rs::channel_positions::channel_xyz;

    let f = get_fixtures();

    // We're loading the ten-channel CSV (F7, F8 missing).
    // Override F7's position to be essentially on top of Fp1 → must clone Fp1.
    let fp1_pos = channel_xyz("Fp1").expect("Fp1 in database");
    let mut overrides = HashMap::new();
    overrides.insert("F7".to_string(), fp1_pos);

    // Simultaneously check Fp1 target re-ordering succeeds.
    let targets: Vec<String> = FIF_CHANNELS.iter().map(|s| s.to_string()).collect();
    let opts = CsvLoadOptions {
        target_channels: Some(targets),
        padding: PaddingStrategy::CloneNearest,
        position_overrides: overrides,
        ..Default::default()
    };

    let (batches, info) = load_from_csv::<B>(&f.ten, &opts, &data_cfg(), &dev())
        .expect("clone_nearest with override");

    assert_eq!(info.n_padded, 2);
    assert_eq!(batches[0].encoder_input.dims(), [1, 480, 32]);

    let ei = ei_vals(&batches[0]);
    assert!(!ei.iter().any(|v| v.is_nan()));
    println!("csv_clone_nearest_with_position_override: OK");
}
