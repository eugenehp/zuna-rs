//! Generate test CSV files from FIF recordings.
//!
//! The raw (unprocessed) EEG data is written as volts so that loading the CSV
//! through the exg pipeline produces numerically identical preprocessed output
//! to loading the same FIF file directly.
//!
//! Format:
//! ```text
//! timestamp,Ch1,Ch2,...
//! 0.000000000e0,1.234567890e-05,...
//! 3.906250000e-3,...
//! ```
//!
//! Values are formatted with `{:.9e}` (10 significant figures), which is more
//! than enough for lossless f32 round-trips.

use std::io::Write as IoWrite;
use std::path::Path;

use exg::fiff::raw::open_raw;

/// Export raw FIF data to a CSV file.
///
/// # Arguments
/// * `fif_path`  – source FIF recording
/// * `csv_path`  – output CSV (created/overwritten)
/// * `keep`      – optional channel subset; if `None` all channels are written.
///                 Names are matched case-insensitively.
///
/// # Returns
/// `(channel_names_written, sample_rate_hz)`
pub fn fif_to_csv(
    fif_path: &Path,
    csv_path: &Path,
    keep: Option<&[&str]>,
) -> (Vec<String>, f32) {
    let raw = open_raw(fif_path).expect("open_raw");
    let sfreq = raw.info.sfreq as f32;
    let all_names: Vec<String> = raw.info.chs.iter().map(|ch| ch.name.clone()).collect();

    // Determine which columns to write
    let keep_indices: Vec<usize> = match keep {
        None => (0..all_names.len()).collect(),
        Some(subset) => {
            let lower: Vec<String> = subset.iter().map(|s| s.to_ascii_lowercase()).collect();
            all_names.iter().enumerate()
                .filter(|(_, n)| lower.contains(&n.to_ascii_lowercase()))
                .map(|(i, _)| i)
                .collect()
        }
    };

    let written_names: Vec<String> = keep_indices.iter()
        .map(|&i| all_names[i].clone())
        .collect();

    // Read raw f64 data [C, T]
    let data_f64 = raw.read_all_data().expect("read_all_data");
    let n_times = data_f64.ncols();

    // Build the output file
    let mut file = std::fs::File::create(csv_path)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", csv_path.display()));

    // Header row
    write!(file, "timestamp").unwrap();
    for name in &written_names {
        write!(file, ",{name}").unwrap();
    }
    writeln!(file).unwrap();

    // Data rows  — cast each value f64 → f32 first (same as load_from_fif)
    let dt = 1.0 / sfreq;
    for t in 0..n_times {
        let ts = t as f32 * dt;
        write!(file, "{ts:.9e}").unwrap();
        for &ci in &keep_indices {
            // f64 → f32 cast matches the cast done inside load_from_fif
            let v = data_f64[[ci, t]] as f32;
            write!(file, ",{v:.9e}").unwrap();
        }
        writeln!(file).unwrap();
    }

    (written_names, sfreq)
}

/// Return the FIF file path used in tests, or panic.
pub fn fif_path() -> std::path::PathBuf {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/sample1_raw.fif");
    assert!(p.exists(), "FIF file not found at {}", p.display());
    p
}
