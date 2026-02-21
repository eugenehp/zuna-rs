//! # zuna-rs — ZUNA EEG Foundation Model inference in Rust
//!
//! Pure-Rust inference for the [ZUNA](https://huggingface.co/Zyphra/ZUNA)
//! EEG foundation model, built on [Burn 0.20](https://burn.dev) and
//! [exg](https://github.com/eugenehp/exg) for FIF preprocessing.
//!
//! ## Three entry points
//!
//! | Type | Loads | Use case |
//! |---|---|---|
//! | [`ZunaInference`] | encoder + decoder | full encode → diffuse → decode pipeline |
//! | [`ZunaEncoder`]   | encoder only      | produce latent embeddings, save memory |
//! | [`ZunaDecoder`]   | decoder only      | reconstruct from stored embeddings |
//!
//! ## Quick start — full pipeline
//!
//! ```rust,ignore
//! use zuna_rs::{ZunaInference, InferenceResult};
//!
//! let (model, _ms) = ZunaInference::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! let result: InferenceResult = model.run_fif(Path::new("recording.fif"), 50, 1.0, 10.0)?;
//! result.save_safetensors("output.safetensors")?;
//! ```
//!
//! ## Quick start — encode only
//!
//! ```rust,ignore
//! use zuna_rs::{ZunaEncoder, EncodingResult};
//!
//! let (enc, _ms) = ZunaEncoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! let result: EncodingResult = enc.encode_fif(Path::new("recording.fif"), 10.0)?;
//! result.save_safetensors("data/embeddings.safetensors")?;
//! ```
//!
//! ## Quick start — decode from stored embeddings
//!
//! ```rust,ignore
//! use zuna_rs::{ZunaDecoder, encoder::EncodingResult};
//!
//! let embeddings = EncodingResult::load_safetensors("data/embeddings.safetensors")?;
//! let (dec, _ms) = ZunaDecoder::<B>::load(
//!     Path::new("config.json"),
//!     Path::new("model.safetensors"),
//!     device,
//! )?;
//! let result = dec.decode_embeddings(&embeddings, 50, 1.0, 10.0)?;
//! result.save_safetensors("output.safetensors")?;
//! ```
//!
//! ## Embedding regularisation
//!
//! The encoder uses an **MMD (Maximum Mean Discrepancy) bottleneck**: during
//! training an MMD loss constrains the embedding distribution toward **N(0, I)**.
//! At inference the bottleneck is a pure passthrough — no reparameterisation is
//! applied.  Embeddings from [`ZunaEncoder`] or [`ZunaInference::encode_fif`]
//! are therefore already in the regularised latent space and can be used
//! directly for downstream tasks.

// ── Internal modules ─────────────────────────────────────────────────────────

pub mod config;
pub mod data;
pub mod encoder;
pub mod decoder;
pub mod inference;
pub mod model;
pub mod weights;

// ── Flat re-exports ───────────────────────────────────────────────────────────
//
// Everything a downstream user needs is available as `zuna_rs::Foo` without
// knowing the internal module layout.

// Full pipeline
pub use inference::{ZunaInference, EpochOutput, InferenceResult};

// Encoder-only
pub use encoder::{ZunaEncoder, EpochEmbedding, EncodingResult};

// Decoder-only
pub use decoder::ZunaDecoder;

// Configs
pub use config::{ModelConfig, DataConfig, InferConfig};

// Data types needed for the lower-level API
pub use data::{InputBatch, FifInfo};
