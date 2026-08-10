# Changelog

All notable changes to `zuna-rs` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2]

### Added

- **`Zyphra/ZUNA1.1` checkpoint support.** Both released checkpoints now load
  through the same API; the architecture is detected from the tensor names, so
  no flag has to be passed. `describe()` and the `infer` banner report which one
  was loaded (`Detected ZUNA1.1 — n_heads = 8`).

  ZUNA1.1 differs from ZUNA1 in exactly three ways, all handled automatically:

  - **QK-norm** — a per-head `RMSNorm(head_dim)` applied to Q and K *before*
    RoPE, in encoder self-attention and in both decoder attentions. Upstream
    hard-codes ε = `1e-5` here rather than using `norm_eps`, and so do we.
  - **Sandwich norm** — an `RMSNorm(dim)` on every sub-layer output *before* the
    residual add: `x + post(sublayer(pre(x)))`.
  - **Key spelling** — ZUNA1.1 wraps each plain RMSNorm in a `torch.nn.RMSNorm`
    sub-module, moving scales from `…ffn_norm.weight` to
    `…ffn_norm.norm.weight`. `canonical_key` collapses these back to the ZUNA1
    spelling so both checkpoints share one key namespace.

  Neither architectural flag appears in `config.json` — upstream hard-codes
  `do_QK_norm` / `do_sandwich_norm` in `lingua/transformer.py` — which is why
  detection reads the weights rather than the config.

- `ModelArch` (with `ZUNA1` / `ZUNA1_1` constants and `detect`) and
  `canonical_key`, re-exported from the crate root.
- `ModelConfig::validate()`, called from every `load()`. It rejects a
  `config.json` that enables a feature this crate does not implement
  (`ape_dim > 0`, `seqlen_t`, `zero_spatial`, a non-`{x,y,z,tc}` `tok_idx_type`,
  or `rope_dim != 4`) instead of silently producing wrong output.
- `ZunaEncoder::arch` / `ZunaDecoder::arch` (both engines) and
  `rlx::ZunaInference::arch()`.
- Tests: a synthetic-checkpoint round trip for both architectures asserting that
  the loader consumes every tensor *and* supplies every parameter the compiled
  graph declares; graph-shape coverage for both variants; a numeric test pinning
  RLX's `rms_norm` semantics on 4-D `[B,S,H,Dh]` tensors; and `ModelConfig` /
  `canonical_key` / `ModelArch` unit tests.
- `rust-version = "1.87"` (1.89 with the optional `burn` backend).
- `BlockDims` / `EncoderDims` / `DecoderDims` geometry structs (each with a
  `from_config` constructor), replacing the long positional argument lists on
  the Burn module constructors.
- `rlx::decoder::RopeInputs`, bundling the `cos`/`sin` buffers passed to
  `ZunaDecoder::forward_step`.
- This changelog.

### Changed

- `load_encoder_weights` / `load_decoder_weights` now return
  `(module, n_heads, ModelArch)` instead of `(module, n_heads)`.
- `EncoderTransformer::new(EncoderDims, device)`,
  `DecoderTransformer::new(DecoderDims, device)`,
  `EncoderDecoder::new(EncoderDims, DecoderDims, global_sigma, device)`,
  `EncoderBlock::new(BlockDims, device)` and
  `DecoderBlock::new(BlockDims, t_dim, device)` now take geometry structs
  instead of a dozen positional `usize`s — a transposed pair of dimensions is
  a compile error rather than a silent shape bug. `Attention::new` and
  `CrossAttention::new` gained a `qk_norm` flag.
- `ZunaDecoder::forward_step` takes `RopeInputs` in place of two slices.
- The whole tree is `cargo fmt`-formatted; `cargo fmt --check`,
  `cargo clippy --all-targets` and `cargo doc` are all clean.
- `rlx::weights::build_encoder_params` / `build_decoder_params` / `load_split`
  return the detected `ModelArch`.
- `rlx::graph::{EncoderSpec, DecoderSpec}` gained an `arch` field.
- `scripts/bench_and_visualize.py`: the NumPy reference encoder handles both
  checkpoints, so the parity and benchmark paths work against either.
- Dependencies moved to the current published releases: `rlx` 0.2.13,
  `exg` 0.0.6.

### Fixed

- **`tests/mlx_op_parity.rs` never compiled.** It was gated on
  `required-features = ["rlx", "mlx"]`, but `mlx` selects the *Burn* MLX
  backend while the test body drives `rlx::Device::Mlx` — so the combination
  was never built and the file silently rotted: it carried an unbalanced `)`
  (a syntax error) and two calls to `Graph::attention_kind_`, a method that no
  longer exists. Gate corrected to `["rlx", "rlx-mlx"]`, syntax fixed, and the
  calls ported to the current `attention_kind(.., shape)` API. All 8 tests in
  it now compile and pass for the first time.
- Module headers across `src/model/`, `src/weights.rs`, `src/config.rs` and the
  binaries used `///` (item docs) instead of `//!` (module docs), so they never
  rendered on docs.rs. Converted, with the ASCII weight-key tables and
  pseudo-code fenced so rustdoc stops reading `[1024, 32]` as a broken link.
  `cargo doc` is now warning-free.
- Stale intra-doc links (`super::data` → `super::rope_helpers`,
  `ZunaEncoder::encode_fif` / `encode_batch` in `decoder.rs`).
- `examples/backend_bench.rs` chose a chart label with an `if` whose two
  branches were identical — dead code, now collapsed (labels unchanged).
- Clippy is clean across all targets: `too_many_arguments` ×11 and
  `type_complexity` ×2 addressed by the structs and type aliases above,
  plus `div_ceil`, `to_vec`, `is_none_or`, `is_multiple_of`, derivable
  `Default`, redundant `into_iter`, and doc-list indentation.
- README stated a minimum of Rust 1.78; the real floor is 1.87.

### Verification

The encoder was checked against an independent NumPy implementation of the
upstream forward pass on `Zyphra/ZUNA1.1` (all 639 tensors, real weights):

| pair | max abs Δ | cosine |
|------|-----------|--------|
| Burn vs NumPy reference | 1.5e-5 | 0.99999994 |
| RLX vs NumPy reference  | 1.4e-5 | 1.00000000 |
| Burn vs RLX             | 2.0e-5 | 0.99999994 |

## [0.2.0]

- Migrated the primary inference path to MLX.

## [0.1.4] – [0.1.0]

- RLX graph backend alongside Burn, multi-backend benchmarks and charts,
  CSV loading, channel-position database.

## [0.0.4] – [0.0.1]

- Initial release: pure-Rust FIF reading and EEG preprocessing via `exg`,
  Burn implementation of the ZUNA encoder/decoder, encoder-only weight loading.
