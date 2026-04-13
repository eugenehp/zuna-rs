/// download_weights — fetch ZUNA model weights from HuggingFace.
///
/// Uses the `hf-hub` crate (same cache layout as Python `huggingface_hub`):
///   ~/.cache/huggingface/hub/models--Zyphra--ZUNA/snapshots/<hash>/…
///
/// A live progress bar is shown during download; already-cached files
/// are returned instantly with no network traffic.
///
/// Usage:
///   download_weights [--repo Zyphra/ZUNA] [--cache-dir ~/.cache/huggingface/hub]
///
/// Prints two lines to stdout (for shell capture):
///   /path/to/model-00001-of-00001.safetensors
///   /path/to/config.json
use anyhow::Result;
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Download ZUNA weights from HuggingFace (no Python required)")]
struct Args {
    /// HuggingFace repo ID.
    #[arg(long, default_value = "Zyphra/ZUNA")]
    repo: String,

    /// Override the HuggingFace cache directory.
    /// Default: $HF_HOME/hub or ~/.cache/huggingface/hub
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

const FILES: &[&str] = &[
    "model-00001-of-00001.safetensors",
    "config.json",
];

fn main() -> Result<()> {
    let args = Args::parse();

    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(dir) = args.cache_dir {
        builder = builder.with_cache_dir(dir);
    }
    let api = builder.build()?;
    let repo = api.model(args.repo);

    for filename in FILES {
        let path = repo.get(filename)?;
        println!("{}", path.display());
    }

    Ok(())
}
