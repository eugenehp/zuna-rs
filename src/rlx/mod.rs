//! RLX-backed implementation of the ZUNA inference pipeline.
//!
//! The legacy Burn-backed types (`crate::ZunaInference`, `crate::ZunaEncoder`,
//! `crate::ZunaDecoder`) are still available at the crate root and remain
//! the reference implementation while the RLX port stabilises. This module
//! re-implements the same architecture on `rlx::Graph` + `rlx::Session` so
//! the two paths can be validated against each other (see
//! `tests/parity_rlx_vs_burn.rs`).
//!
//! Enable with `--features rlx-backend`.

pub mod data;
pub mod decoder;
pub mod encoder;
pub mod graph;
pub mod inference;
pub mod weights;

pub use decoder::{EpochOutput, ZunaDecoder};
pub use encoder::{EpochEmbedding, ZunaEncoder};
pub use inference::{InferenceResult, ZunaInference};
