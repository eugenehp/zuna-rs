//! # zuna-rs — ZUNA EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the [ZUNA](https://huggingface.co/Zyphra/ZUNA)
//! EEG foundation model. Two inference engines are available behind Cargo
//! features:
//!
//! | feature | module | runtime |
//! |---------|--------|---------|
//! | `burn`  | crate root (`ZunaEncoder`, …) | [Burn](https://burn.dev) 0.20 |
//! | `rlx`   | [`rlx`] | [RLX](https://docs.rs/rlx) compiler/runtime |
//!
//! FIF preprocessing is shared ([`preprocess_fif_cpu`]) via [exg](https://github.com/eugenehp/exg).
//!
//! ## Backends
//!
//! **Burn** (with `--features burn,ndarray`): `ndarray`, `blas-accelerate`,
//! `wgpu`, `burn-mlx` / `mlx`.
//!
//! **RLX** (with `--features rlx`): `cpu`, `metal`, `mlx`, `gpu`, `cuda`,
//! `rocm`, `tpu`, and BLAS variants.
//!
//! Compare both engines (add `--features burn,ndarray` for the Burn side):
//!
//! ```text
//! cargo run --example backend_compare --release \
//!     --no-default-features \
//!     --features burn,rlx,ndarray,rlx-cpu,rlx-metal,metal,wgpu,mlx,rlx-mlx
//! ```

// At least one inference engine must be enabled.
#[cfg(not(any(feature = "burn", feature = "rlx")))]
compile_error!("enable at least one inference engine: `rlx` (default) and/or `burn`");

/// Configure the global Rayon thread pool (Burn NdArray + RLX CPU).
pub fn init_threads(n: Option<usize>) -> usize {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if let Some(count) = n {
        if count > 0 {
            builder = builder.num_threads(count);
        }
    }
    let _ = builder.build_global();
    rayon::current_num_threads()
}

pub mod channel_positions;
pub mod config;
pub mod csv_export;
pub mod csv_loader;
pub mod data;

#[cfg(feature = "burn")]
pub mod model;

#[cfg(feature = "burn")]
pub mod encoder;

#[cfg(feature = "burn")]
pub mod decoder;

#[cfg(feature = "burn")]
pub mod inference;

#[cfg(feature = "burn")]
pub mod weights;

#[cfg(feature = "rlx")]
pub mod rlx;

// ── Burn re-exports (crate root) ─────────────────────────────────────────────

#[cfg(feature = "burn")]
pub use inference::{InferenceResult, ZunaInference};

#[cfg(feature = "burn")]
pub use encoder::{EpochEmbedding, EncodingResult, ZunaEncoder};

#[cfg(feature = "burn")]
pub use decoder::ZunaDecoder;

// When Burn is off, lift the RLX API to the crate root (default build).
#[cfg(all(feature = "rlx", not(feature = "burn")))]
pub use rlx::{
    EpochEmbedding, EpochOutput, InferenceResult, ZunaDecoder, ZunaEncoder, ZunaInference,
};

// ── Shared types ───────────────────────────────────────────────────────────

pub use config::{DataConfig, InferConfig, ModelConfig};

pub use data::{
    FifInfo, PreprocessedEpoch, PreprocessedFif, invert_reshape, preprocess_fif_cpu,
};

#[cfg(feature = "burn")]
pub use data::{InputBatch, preprocessed_to_batch};

pub use channel_positions::{
    MontageLayout, channel_xyz, montage_channels, nearest_channel, normalise,
};

pub use csv_loader::{
    CsvInfo, CsvLoadOptions, PaddingStrategy, load_from_csv, load_from_named_tensor,
    load_from_raw_tensor,
};

pub use csv_export::fif_to_csv;
