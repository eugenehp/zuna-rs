#!/usr/bin/env python3
"""
dump_python_reference.py — emit a Python-side ZUNA encoder reference
====================================================================

Runs the NumPy reimplementation of the ZUNA encoder (from
``bench_and_visualize.py``) on a single FIF epoch and saves
``{encoder_input, tok_idx, embedding}`` to a safetensors file so the
Rust parity test (``tests/parity_rlx_vs_python.rs``) can compare against
it.

The Rust preprocessing pipeline is NOT used here; instead we rely on
the Rust binary to dump pre-tokenised inputs (``cargo run --release \
    --bin infer -- --export-inputs ...``) or, lacking that, we read a
pre-existing inputs safetensors file. For simplicity this script
expects the inputs file to be passed in.

Usage
-----
    # 1. Get the ZUNA weights (downloads on first use)
    cargo run --release --bin download_weights --features hf-download

    # 2. Use the Rust preprocessing once to dump tokenised inputs:
    #    (any of the existing examples that writes encoder_input_N / tok_idx_N
    #    keys to a safetensors file works — channel_bench does this.)
    cargo run --release --example channel_bench

    # 3. Run this script with the printed weights path and the inputs file:
    python3 scripts/dump_python_reference.py \
        --weights ~/.cache/huggingface/hub/.../model.safetensors \
        --config  ~/.cache/huggingface/hub/.../config.json \
        --inputs  data/encoder_inputs.safetensors \
        --output  data/python_reference.safetensors

The output file contains, for one chosen epoch (default: epoch 0):
    encoder_input  [S, 32]   float32  — same buffer Rust ingested
    tok_idx        [S, 4]    int32    — same indices Rust used
    embedding      [S, 32]   float32  — Python NumPy encoder output
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Reuse the NumPy encoder + loader from the benchmark script.
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from bench_and_visualize import (  # noqa: E402
    ZunaEncoderNumpy,
    load_weights_f32,
    load_safetensors_numpy,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=pathlib.Path,
                    help="Path to ZUNA model.safetensors")
    ap.add_argument("--config",  required=True, type=pathlib.Path,
                    help="Path to ZUNA config.json")
    ap.add_argument("--inputs",  required=True, type=pathlib.Path,
                    help="Pre-tokenised encoder inputs (safetensors with "
                         "encoder_input_N / tok_idx_N / chan_pos_N keys)")
    ap.add_argument("--output",  required=True, type=pathlib.Path,
                    help="Output safetensors file")
    ap.add_argument("--epoch",   type=int, default=0,
                    help="Epoch index to dump (default 0)")
    args = ap.parse_args()

    with args.config.open() as fh:
        config = json.load(fh)

    print(f"→ Loading weights from {args.weights}")
    weights = load_weights_f32(args.weights)

    print(f"→ Loading inputs from {args.inputs}")
    inputs = load_safetensors_numpy(args.inputs)
    enc_input_key = f"encoder_input_{args.epoch}"
    tok_idx_key   = f"tok_idx_{args.epoch}"
    if enc_input_key not in inputs or tok_idx_key not in inputs:
        print(f"  ✗ Missing keys for epoch {args.epoch}", file=sys.stderr)
        print(f"    expected: {enc_input_key}, {tok_idx_key}", file=sys.stderr)
        print(f"    have:     {sorted(inputs.keys())[:8]} ...", file=sys.stderr)
        return 1

    enc_input = inputs[enc_input_key].astype(np.float32)
    if enc_input.ndim == 2:
        enc_input = enc_input[np.newaxis]  # → [1, S, 32]
    tok_idx = inputs[tok_idx_key].astype(np.int32)

    s = enc_input.shape[1]
    print(f"→ Encoder input  [1, {s}, {enc_input.shape[2]}]")
    print(f"→ Tok idx        [{tok_idx.shape[0]}, {tok_idx.shape[1]}]")

    encoder = ZunaEncoderNumpy(weights, config)
    embedding = encoder.forward(enc_input, tok_idx.astype(np.int64))
    print(f"→ Embedding      [{embedding.shape[0]}, {embedding.shape[1]}]")

    out_dict = {
        "encoder_input": enc_input[0],            # [S, 32]
        "tok_idx":       tok_idx,                 # [S, 4]
        "embedding":     embedding.astype(np.float32),  # [S, 32]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_dict, str(args.output))
    print(f"✓ Wrote {args.output}")
    print(f"  Keys: {list(out_dict.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
