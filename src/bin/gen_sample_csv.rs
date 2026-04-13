//! Generate `data/sample.csv` from `data/sample1_raw.fif`.
//!
//! Run once before using `csv_embed` or any other CSV-based example:
//!
//! ```sh
//! cargo run --bin gen_sample_csv
//! ```
//!
//! Options:
//! ```sh
//! cargo run --bin gen_sample_csv -- \
//!     --fif  data/sample1_raw.fif \
//!     --csv  data/sample.csv \
//!     --channels "Fp1,Fp2,F3,F4"   # omit to write all channels
//! ```

use std::path::PathBuf;
use clap::Parser;
use zuna_rs::fif_to_csv;

#[derive(Parser, Debug)]
#[command(
    name  = "gen_sample_csv",
    about = "Export a FIF recording to the CSV format expected by csv_embed and csv_loader",
)]
struct Args {
    /// Source FIF file.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample1_raw.fif"))]
    fif: PathBuf,

    /// Output CSV path.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data/sample.csv"))]
    csv: PathBuf,

    /// Comma-separated channel names to include (default: all channels).
    #[arg(long)]
    channels: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let keep_owned: Option<Vec<String>> = args.channels.as_deref().map(|s| {
        s.split(',').map(|c| c.trim().to_string()).collect()
    });
    let keep_refs: Option<Vec<&str>> = keep_owned.as_deref().map(|v| {
        v.iter().map(|s| s.as_str()).collect()
    });

    println!("Source FIF : {}", args.fif.display());
    println!("Output CSV : {}", args.csv.display());
    if let Some(ref k) = keep_owned {
        println!("Channels   : {}", k.join(", "));
    } else {
        println!("Channels   : all");
    }

    let (names, sfreq) = fif_to_csv(
        &args.fif,
        &args.csv,
        keep_refs.as_deref(),
    )?;

    println!("\nWrote {} channels × {} Hz → {}",
        names.len(), sfreq, args.csv.display());
    println!("Channels   : {}", names.join(", "));
    Ok(())
}
