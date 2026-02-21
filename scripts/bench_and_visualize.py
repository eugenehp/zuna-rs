#!/usr/bin/env python3
# Works with any Python 3 interpreter name:
#   python3 bench_and_visualize.py ...   (Linux, most macOS)
#   python  bench_and_visualize.py ...   (conda/Windows where 'python3' is absent)
#   ./bench_and_visualize.py ...         (requires chmod +x and python3 in PATH)
"""
bench_and_visualize.py — ZUNA Python-NumPy vs Rust benchmark
=============================================================

Compares the ZUNA EEG encoder across implementations:

  Python  — NumPy re-implementation of the encoder transformer forward pass,
             fed the exact same pre-tokenised tensors produced by Rust/exg.
  Rust    — `cargo run --example embed --release` (CPU NdArray backend).

Metrics collected
-----------------
  Speed       : wall-clock time for each pipeline phase (N independent runs).
  Precision   : element-wise MAE, RMSE, max-error, Pearson r between
                Python and Rust encoder outputs on identical inputs.
  Distribution: histogram of embedding values vs N(0,1) (MMD regularlisation).
  Variance     : per-dimension mean ± std, per-run consistency (std across runs).

Outputs
-------
  All filenames include a platform slug (e.g. cpu_apple_m3_pro or
  gpu_nvidia_rtx_4090) so results from different machines never collide.

  figures/bench_speed_<slug>.png            Per-phase wall-clock comparison chart.
  figures/bench_py_vs_rust_<slug>.png       Python vs Rust error distribution.
  figures/bench_precision_<slug>.png        Error metrics bar chart.
  figures/bench_distribution_<slug>.png     Embedding histograms vs N(0,1).
  figures/bench_dim_stats_<slug>.png        Per-dimension mean/std heatmap.
  figures/bench_run_consistency_<slug>.png  Across-run variance (Rust).
  figures/bench_data_<slug>.json            All raw benchmark numbers + platform info.
  README.md                                 "Benchmark" section auto-updated.

Requirements: numpy, safetensors, matplotlib, huggingface_hub
Optional:     scipy (for KL-divergence), tqdm (nicer progress)

Usage
-----
  # Works with python or python3:
  python  bench_and_visualize.py --runs 3 --no-python-encoder
  python3 bench_and_visualize.py --runs 5
  python  bench_and_visualize.py --weights model.safetensors --config config.json
"""

from __future__ import annotations
import argparse
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

# Propagate the running interpreter to Rust subprocesses.
# common/mod.rs find_python() checks $ZUNA_PYTHON first, so this ensures
# any 'cargo run --example embed' (or pre-built binary) spawned from here
# uses the same Python that launched this script — critical on macOS where
# 'python3' may be absent but 'python' (conda/brew) is available.
os.environ.setdefault("ZUNA_PYTHON", sys.executable)

try:
    import numpy as np
except ModuleNotFoundError:
    # Re-exec under the alternative interpreter (python3 → python or vice-versa)
    # before giving up, so the script works when only one of them has numpy.
    import os as _os, sys as _sys, shutil as _sh
    _base = _os.path.basename(_sys.executable)
    if not _os.environ.get("_ZUNA_PYTHON_RETRIED"):
        _alt = "python" if "3" in _base else "python3"
        _alt_path = _sh.which(_alt)
        if _alt_path:
            _os.execve(
                _alt_path,
                [_alt_path] + _sys.argv,
                {**_os.environ, "_ZUNA_PYTHON_RETRIED": "1"},
            )
    _sys.exit(
        f"✗  numpy not found (tried python3 and python).\n"
        f"   Install: {_sys.executable} -m pip install numpy"
    )

# ── Optional dependencies ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠  matplotlib not found — charts will be skipped (pip install matplotlib)")

try:
    from safetensors.numpy import load_file as _st_load
    from safetensors import safe_open
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("⚠  safetensors not found — Python encoder comparison will be skipped")

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("⚠  huggingface_hub not found (pip install huggingface_hub)")

try:
    from scipy.stats import entropy as kl_entropy
    from scipy.special import rel_entr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ID      = "Zyphra/ZUNA"
WEIGHTS_FILE = "model-00001-of-00001.safetensors"
CONFIG_FILE  = "config.json"
SAMPLE_FIF   = pathlib.Path(__file__).parent / "data" / "sample1_raw.fif"
FIGURES_DIR  = pathlib.Path(__file__).parent / "figures"
BENCH_DATA   = FIGURES_DIR / "bench_data.json"
README_PATH  = pathlib.Path(__file__).parent / "README.md"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# ── Platform detection ────────────────────────────────────────────────────────

def _sysctl(key: str) -> str:
    """Return `sysctl -n <key>` output stripped, or '' on any error."""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


# Theoretical peak fp32 TFLOPS for common chips (CPU/GPU combined for Apple
# Silicon; GPU-only figures for NVIDIA discrete cards).
_TFLOPS_TABLE: Dict[str, float] = {
    # Apple Silicon (GPU die, fp32)
    "Apple M1":          2.6,  "Apple M1 Pro":   5.2,
    "Apple M1 Max":     10.4,  "Apple M1 Ultra": 20.8,
    "Apple M2":          3.6,  "Apple M2 Pro":    6.8,
    "Apple M2 Max":     13.6,  "Apple M2 Ultra":  27.2,
    "Apple M3":          3.6,  "Apple M3 Pro":    7.2,
    "Apple M3 Max":     14.2,  "Apple M3 Ultra":  28.4,
    "Apple M4":          3.8,  "Apple M4 Pro":    9.0,
    "Apple M4 Max":     18.0,  "Apple M4 Ultra":  36.0,
    # NVIDIA discrete (fp32, boost clock)
    "H200":  67.0,  "H100":  67.0,  "A100":  19.5,
    "L40S":  91.6,  "L40":   45.3,  "A40":   37.4,
    "RTX 4090": 82.6,  "RTX 4080": 48.7,  "RTX 4070 Ti": 40.1,
    "RTX 3090": 35.6,  "RTX 3080": 29.8,  "RTX 3070":  20.3,
    "RTX 2080 Ti": 13.4,  "T4": 8.1,  "V100": 14.1,
}


def collect_platform_info(device_type: str = "cpu") -> dict:
    """
    Collect OS, CPU, GPU, and memory characteristics.

    Returns a dict with keys:
      os, os_version, os_release, arch, python_version,
      device_type, cpu_model, cpu_cores_physical, cpu_cores_logical,
      ram_gb, gpu_model, gpu_vram_gb, estimated_tflops,
      slug   (filename-safe short id, e.g. 'cpu_apple_m3_pro'),
      label  (human-readable one-liner for chart footers).
    """
    import platform as _platform
    system = _platform.system()

    info: dict = {
        "os":                 system,
        "os_version":         _platform.version(),
        "os_release":         _platform.release(),
        "arch":               _platform.machine(),
        "python_version":     _platform.python_version(),
        "device_type":        device_type,
        "cpu_model":          "unknown",
        "cpu_cores_physical": None,
        "cpu_cores_logical":  os.cpu_count() or 1,
        "ram_gb":             None,
        "gpu_model":          None,
        "gpu_vram_gb":        None,
        "estimated_tflops":   None,
    }

    # ── macOS ─────────────────────────────────────────────────────────────
    if system == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")   # works on Intel
        if brand:
            info["cpu_model"] = brand
        else:
            # Apple Silicon: system_profiler carries the chip name
            try:
                sp = subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType"],
                    stderr=subprocess.DEVNULL, text=True,
                )
                m = re.search(r"(?:Chip|Processor Name):\s*(.+)", sp)
                if m:
                    info["cpu_model"] = m.group(1).strip()
            except Exception:
                pass

        pc = _sysctl("hw.physicalcpu")
        lc = _sysctl("hw.logicalcpu")
        if pc:
            info["cpu_cores_physical"] = int(pc)
        if lc:
            info["cpu_cores_logical"] = int(lc)

        mem = _sysctl("hw.memsize")
        if mem:
            info["ram_gb"] = round(int(mem) / 1024 ** 3, 1)

        # Apple Silicon: GPU is on the same die; unified memory = VRAM
        if "Apple" in info["cpu_model"] and device_type == "gpu":
            info["gpu_model"]   = info["cpu_model"] + " (integrated GPU / Metal)"
            info["gpu_vram_gb"] = info["ram_gb"]

    # ── Linux ─────────────────────────────────────────────────────────────
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as _f:
                for _line in _f:
                    if _line.startswith("model name"):
                        info["cpu_model"] = _line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

        try:
            with open("/proc/meminfo") as _f:
                for _line in _f:
                    if _line.startswith("MemTotal"):
                        info["ram_gb"] = round(int(_line.split()[1]) / 1024 ** 2, 1)
                        break
        except Exception:
            pass

        # Physical core count via unique (physical_id, core_id) pairs
        try:
            with open("/proc/cpuinfo") as _f:
                _text = _f.read()
            _cores: set = set()
            for _block in _text.split("\n\n"):
                _pm = re.search(r"physical id\s*:\s*(\d+)", _block)
                _cm = re.search(r"core id\s*:\s*(\d+)",     _block)
                if _pm and _cm:
                    _cores.add((_pm.group(1), _cm.group(1)))
            if _cores:
                info["cpu_cores_physical"] = len(_cores)
        except Exception:
            pass

        # NVIDIA GPU via nvidia-smi
        try:
            _smi = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,clocks.max.sm",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip().splitlines()[0]
            _parts = [p.strip() for p in _smi.split(",")]
            info["gpu_model"] = _parts[0] if _parts else None
            if len(_parts) > 1 and _parts[1].isdigit():
                info["gpu_vram_gb"] = round(int(_parts[1]) / 1024, 1)
        except Exception:
            pass

    # ── TFLOPS lookup (works for any OS) ──────────────────────────────────
    # Sort longest key first so "Apple M3 Max" matches before "Apple M3".
    _target = (info.get("gpu_model") or info["cpu_model"]).lower()
    for _chip, _tflops in sorted(_TFLOPS_TABLE.items(), key=lambda kv: -len(kv[0])):
        if _chip.lower() in _target:
            info["estimated_tflops"] = _tflops
            break

    # ── Slug (filename-safe, ≤ 36 chars) ──────────────────────────────────
    # Use GPU model when running on GPU, otherwise CPU model.
    # Strip parentheticals like "(integrated GPU / Metal)" before slugifying.
    _slug_base = (info["gpu_model"] if device_type == "gpu" and info["gpu_model"]
                  else info["cpu_model"])
    _slug_base = re.sub(r"\s*\(.*?\)", "", _slug_base).strip()
    _raw = _slug_base.lower()
    _raw = re.sub(r"[^a-z0-9]+", "_", _raw).strip("_")
    _raw = re.sub(r"_+", "_", _raw)
    if len(_raw) > 32:
        _raw = _raw[:32].rstrip("_")
    info["slug"] = f"{device_type}_{_raw}"

    # ── Human-readable label ───────────────────────────────────────────────
    _parts: List[str] = [f"{system} {info['os_release']}", info["cpu_model"]]
    _pc = info["cpu_cores_physical"]
    _lc = info["cpu_cores_logical"]
    if _pc and _lc and _pc != _lc:
        _parts.append(f"{_pc}P / {_lc}L cores")
    elif _lc:
        _parts.append(f"{_lc} cores")
    if info["ram_gb"]:
        _parts.append(f"{info['ram_gb']} GB RAM")
    if device_type == "gpu" and info["gpu_model"]:
        _parts.append(f"GPU: {info['gpu_model']}")
        if info["gpu_vram_gb"]:
            _parts.append(f"{info['gpu_vram_gb']} GB VRAM")
    if info["estimated_tflops"]:
        _parts.append(f"~{info['estimated_tflops']} TFLOPS (fp32)")
    info["label"] = " · ".join(_parts)

    return info


# ── Weight / config loading ───────────────────────────────────────────────────

def default_hf_cache() -> pathlib.Path:
    if v := os.environ.get("HF_HOME"):
        return pathlib.Path(v) / "hub"
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or "."
    return pathlib.Path(home) / ".cache" / "huggingface" / "hub"


def find_snapshot(repo_id: str, cache_dir: Optional[pathlib.Path] = None) -> Optional[pathlib.Path]:
    """Return the newest snapshot path for repo_id in the HF cache, or None."""
    base = cache_dir or default_hf_cache()
    snap_root = base / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snap_root.exists():
        return None
    snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime)
    return snaps[-1] if snaps else None


def resolve_weights(
    repo_id: str,
    weights: Optional[str],
    config:  Optional[str],
    cache_dir: Optional[pathlib.Path] = None,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Return (weights_path, config_path), downloading if needed."""
    if weights and config:
        return pathlib.Path(weights), pathlib.Path(config)

    # Try HF cache first
    snap = find_snapshot(repo_id, cache_dir)
    if snap:
        w = snap / WEIGHTS_FILE
        c = snap / CONFIG_FILE
        if w.exists() and c.exists():
            print(f"  ✓ Using cached snapshot: {snap}")
            return w, c

    # Download
    if HAS_HF:
        print(f"  Downloading {repo_id} from HuggingFace …")
        snap_path = snapshot_download(repo_id, local_files_only=False)
        snap = pathlib.Path(snap_path)
        return snap / WEIGHTS_FILE, snap / CONFIG_FILE
    else:
        raise RuntimeError(
            f"Model not in cache and huggingface_hub not installed.\n"
            f"Install with: pip install huggingface_hub\n"
            f"Then run: {sys.executable} -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{repo_id}')\""
        )


def load_config(config_path: pathlib.Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_weights_f32(weights_path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load safetensors file as float32 numpy arrays (handles bfloat16)."""
    if not HAS_ST:
        raise RuntimeError("safetensors not installed")

    weights: Dict[str, np.ndarray] = {}
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            k = key[len("model."):] if key.startswith("model.") else key
            try:
                arr = f.get_tensor(key)
            except Exception as e:
                print(f"  ⚠ Could not load {key}: {e}")
                continue

            # Handle BF16 → F32 conversion
            # safetensors numpy returns uint16 for BF16 (numpy has no BF16 dtype)
            if arr.dtype in (np.uint16,) or (arr.dtype.itemsize == 2
                    and not np.issubdtype(arr.dtype, np.floating)):
                # BF16: the 16-bit value occupies the UPPER 16 bits of float32
                # Shift left by 16 gives the float32 bit pattern
                u16 = arr.view(np.uint16)
                weights[k] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(arr.shape)
            elif str(arr.dtype) in ("bfloat16", "bf16"):
                # ml_dtypes bfloat16 — just cast
                weights[k] = arr.astype(np.float32)
            else:
                weights[k] = arr.astype(np.float32)

    print(f"  Loaded {len(weights)} weight tensors from {weights_path.name}")
    return weights


# ── NumPy ZUNA encoder implementation ─────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class ZunaEncoderNumpy:
    """
    Pure-NumPy implementation of the ZUNA encoder transformer.

    Accepts pre-tokenised data (encoder_input [1,S,32] + tok_idx [S,4])
    produced by the Rust preprocessing pipeline (exg) and saved via
    `cargo run --example embed -- --export-inputs`.

    The forward pass exactly mirrors the Burn/Rust implementation:
      1. Interleave register tokens   [1, 2S, 32]
      2. Embed with tok_embeddings    [1, 2S, dim]
      3. 16× TransformerBlock (RMSNorm + Self-Attention(4D-RoPE) + SwiGLU)
      4. Extract register tokens      [1, S, dim]
      5. RMSNorm + output projection  [1, S, 32]
    """

    def __init__(self, weights: Dict[str, np.ndarray], config: dict):
        self.w   = weights
        cfg      = config["model"]
        self.n_layers   = cfg["n_layers"]
        self.head_dim   = cfg["head_dim"]
        self.dim        = cfg["dim"]
        self.input_dim  = cfg["input_dim"]
        self.output_dim = cfg["encoder_output_dim"]
        self.rope_dim   = cfg["rope_dim"]
        self.max_seqlen = cfg["max_seqlen"]
        self.rope_theta = cfg["rope_theta"]
        self.norm_eps   = float(cfg.get("norm_eps", 1e-5))

        # n_heads is NOT stored in config.json — infer from weight shapes
        # wq.weight: [n_heads * head_dim, dim]
        wq_key = "encoder.layers.0.attention.wq.weight"
        if wq_key in weights:
            self.n_heads = weights[wq_key].shape[0] // self.head_dim
        else:
            self.n_heads = self.dim // self.head_dim   # fallback: 1024//64 = 16
        self.n_kv_heads = self.n_heads   # ZUNA uses full MHA (no GQA)

        # Precompute RoPE table: [max_seqlen, head_dim/(rope_dim*2), 2, 2]
        dim_per_rope  = self.head_dim // self.rope_dim   # 16
        half_per_axis = dim_per_rope  // 2               # 8
        table = np.zeros((self.max_seqlen, half_per_axis, 2, 2), dtype=np.float32)
        for pos in range(self.max_seqlen):
            for h in range(half_per_axis):
                freq  = 1.0 / (self.rope_theta ** (2 * h / dim_per_rope))
                angle = pos * freq
                c, s  = math.cos(angle), math.sin(angle)
                table[pos, h] = [[c, -s], [s, c]]
        self._freqs_cis = table   # [max_seqlen, 8, 2, 2]

    # ── RoPE helpers ─────────────────────────────────────────────────────────

    def _build_freqs_4d(self, tok_idx_2x: np.ndarray) -> np.ndarray:
        """[2S, 4] → [2S, head_dim//2, 2, 2] (concatenated per axis)."""
        parts = []
        for axis in range(self.rope_dim):
            idx = np.clip(tok_idx_2x[:, axis].astype(np.int64), 0, self.max_seqlen - 1)
            parts.append(self._freqs_cis[idx])       # [2S, 8, 2, 2]
        return np.concatenate(parts, axis=1)          # [2S, 32, 2, 2]

    @staticmethod
    def _apply_rope(xq: np.ndarray, xk: np.ndarray,
                    freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        xq, xk  : [B, S, H, D]
        freqs    : [S, D//2, 2, 2]
        Returns  : rotated (xq', xk') same shape.

        Matches Rust `rotate_half` exactly:
          pairs  = reshape [B,S,H,D] → [B,S,H,D//2, 2]
          cos    = freqs[..., 0, 0]     [S, D//2]
          sin    = freqs[..., 1, 0]     [S, D//2]
          even'  = even*cos  − odd*sin
          odd'   = even*sin  + odd*cos
          result = stack([even', odd'], axis=-1).reshape([B,S,H,D])
        """
        B, S, H, D = xq.shape
        half = D // 2
        cos = freqs[:, :, 0, 0][np.newaxis, :, np.newaxis, :]  # [1,S,1,D/2]
        sin = freqs[:, :, 1, 0][np.newaxis, :, np.newaxis, :]  # [1,S,1,D/2]

        def rotate(x: np.ndarray) -> np.ndarray:
            pairs    = x.reshape(B, S, H, half, 2)
            even, odd = pairs[..., 0], pairs[..., 1]
            e_out    = even * cos - odd * sin
            o_out    = even * sin + odd * cos
            return np.stack([e_out, o_out], axis=-1).reshape(B, S, H, D)

        return rotate(xq), rotate(xk)

    # ── Layer building blocks ─────────────────────────────────────────────────

    def _rms_norm(self, x: np.ndarray, w_key: str) -> np.ndarray:
        """RMSNorm along last dim.
        w_key refers to self.w[w_key] which holds the RMSNorm scale vector.
        In the safetensors file: 'encoder.layers.i.attention_norm.weight' etc.
        """
        gamma = self.w[w_key]
        rms   = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.norm_eps)
        return (x / rms) * gamma

    def _self_attention(self, x: np.ndarray, layer: int,
                        freqs: np.ndarray) -> np.ndarray:
        """[1, S, dim] → [1, S, dim]  (full MHA with 4D RoPE)."""
        p   = f"encoder.layers.{layer}.attention"
        wq  = self.w[f"{p}.wq.weight"]   # [dim, dim]
        wk  = self.w[f"{p}.wk.weight"]
        wv  = self.w[f"{p}.wv.weight"]
        wo  = self.w[f"{p}.wo.weight"]

        B, S, _ = x.shape
        H, Dh   = self.n_heads, self.head_dim

        xq = (x @ wq.T).reshape(B, S, H, Dh)
        xk = (x @ wk.T).reshape(B, S, H, Dh)
        xv = (x @ wv.T).reshape(B, S, H, Dh)

        xq, xk = self._apply_rope(xq, xk, freqs)

        # [1, H, S, Dh]
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        scale  = Dh ** -0.5
        scores = xq @ xk.transpose(0, 1, 3, 2) * scale   # [1, H, S, S]
        attn   = softmax(scores, axis=-1)
        out    = attn @ xv                                  # [1, H, S, Dh]

        out = out.transpose(0, 2, 1, 3).reshape(B, S, H * Dh)
        return out @ wo.T

    def _swiglu(self, x: np.ndarray, layer: int) -> np.ndarray:
        """SwiGLU feed-forward: w2(silu(w1(x)) * w3(x))."""
        p  = f"encoder.layers.{layer}.feed_forward"
        w1 = self.w[f"{p}.w1.weight"]   # [hidden, dim]
        w2 = self.w[f"{p}.w2.weight"]   # [dim, hidden]
        w3 = self.w[f"{p}.w3.weight"]
        return silu(x @ w1.T) * (x @ w3.T) @ w2.T

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, encoder_input: np.ndarray,
                tok_idx: np.ndarray) -> np.ndarray:
        """
        encoder_input : [1, S, 32]  float32  (pre-tokenised EEG)
        tok_idx       : [S, 4]      int64    (discrete x,y,z,tc positions)
        Returns       : [S, output_dim]  float32  (encoder latent embedding)
        """
        B, S, D = encoder_input.shape
        w = self.w

        # ── 1. Interleave register tokens [1, S, 32] + [1, S, 32] → [1, 2S, 32] ──
        # "encoder.registers" shape is (1, 32) in safetensors
        regs_proto  = w["encoder.registers"]            # (1, 32)
        regs        = np.broadcast_to(regs_proto[np.newaxis], (B, S, D)).copy()
        interleaved = np.stack([regs, encoder_input], axis=2).reshape(B, S * 2, D)

        # ── 2. Embed → [1, 2S, dim] ───────────────────────────────────────────
        emb_w = w["encoder.tok_embeddings.weight"]          # [dim, 32]
        emb_b = w.get("encoder.tok_embeddings.bias")        # [dim] or None
        h = interleaved @ emb_w.T
        if emb_b is not None:
            h = h + emb_b

        # ── 3. Repeat tok_idx: [S, 4] → [2S, 4] ──────────────────────────────
        tok_idx_2x = np.repeat(tok_idx, 2, axis=0)

        # ── 4. Build 4D RoPE freqs [2S, head_dim//2, 2, 2] ───────────────────
        freqs_4d = self._build_freqs_4d(tok_idx_2x)         # [2S, 32, 2, 2]

        # ── 5. Transformer layers ─────────────────────────────────────────────
        # NOTE: safetensors key pattern is 'encoder.layers.i.attention_norm.weight'
        #       (NOT '.inner.gamma' which is the Burn module path, not the file key)
        for i in range(self.n_layers):
            p = f"encoder.layers.{i}"

            # Attention sub-layer
            h_n = self._rms_norm(h, f"{p}.attention_norm.weight")
            h   = h + self._self_attention(h_n, i, freqs_4d)

            # Feed-forward sub-layer
            h_n = self._rms_norm(h, f"{p}.ffn_norm.weight")
            h   = h + self._swiglu(h_n, i)

        # ── 6. Extract register tokens (even positions 0,2,4,…) ──────────────
        hdim = h.shape[-1]
        regs_out = h.reshape(B, S, 2, hdim)[:, :, 0, :]     # [1, S, dim]

        # ── 7. RMSNorm + output projection ────────────────────────────────────
        out = self._rms_norm(regs_out, "encoder.norm.weight")
        out = out @ w["encoder.output.weight"].T             # [1, S, output_dim]

        return out.squeeze(0)                                # [S, output_dim]


# ── Rust runner ───────────────────────────────────────────────────────────────

def _parse_timing_line(stderr: str) -> Optional[dict]:
    """Parse 'TIMING weights=Xms preproc=Xms encode=Xms total=Xms' from stderr."""
    m = re.search(
        r"TIMING\s+weights=(\S+)ms\s+preproc=(\S+)ms\s+encode=(\S+)ms\s+total=(\S+)ms",
        stderr
    )
    if not m:
        return None
    return {
        "weights_ms": float(m.group(1)),
        "preproc_ms": float(m.group(2)),
        "encode_ms":  float(m.group(3)),
        "total_ms":   float(m.group(4)),
    }


def run_rust_embed(
    weights_path: pathlib.Path,
    config_path:  pathlib.Path,
    fif_path:     pathlib.Path,
    output_path:  pathlib.Path,
    export_inputs_path: Optional[pathlib.Path] = None,
    verbose: bool = False,
    embed_bin: Optional[pathlib.Path] = None,
    device: str = "cpu",
) -> Tuple[Optional[dict], float]:
    """
    Run the embed example and return (timing_dict, wall_ms).

    If embed_bin is given, run it directly (pre-built binary — no compile wait).
    Otherwise fall back to `cargo run --example embed --release`.
    The cargo build phase is excluded from wall_ms when using `cargo run`.
    """
    if embed_bin and pathlib.Path(embed_bin).is_file():
        cmd = [str(embed_bin)]
    else:
        cargo = shutil.which("cargo") or "cargo"
        cmd   = [cargo, "run", "--example", "embed", "--release", "--quiet", "--"]

    cmd += [
        "--weights", str(weights_path),
        "--config",  str(config_path),
        "--fif",     str(fif_path),
        "--output",  str(output_path),
        "--device",  device,
        "--no-charts",
    ]
    if export_inputs_path:
        cmd += ["--export-inputs", str(export_inputs_path)]
    if verbose:
        cmd.append("--verbose")

    t0     = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=pathlib.Path(__file__).parent)
    wall   = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        print("  ⚠ Rust embed failed:")
        print("  stdout:", result.stdout[-400:])
        print("  stderr:", result.stderr[-400:])
        return None, wall

    timing = _parse_timing_line(result.stderr)
    return timing, wall


def run_rust_benchmark(
    weights_path: pathlib.Path,
    config_path:  pathlib.Path,
    fif_path:     pathlib.Path,
    n_runs:       int,
    tmp_dir:      pathlib.Path,
    export_inputs_on_run: int = 0,
    device: str = "cpu",
    embed_bin: Optional[pathlib.Path] = None,
) -> Tuple[List[dict], Optional[pathlib.Path]]:
    """
    Run Rust embed N times for timing stats.
    On run `export_inputs_on_run`, also export encoder inputs.
    Returns (timing_list, inputs_safetensors_path_or_None).
    """
    timings: List[dict] = []
    inputs_path: Optional[pathlib.Path] = None

    print(f"\n▶ Running Rust encoder {n_runs}× for timing benchmarks …")
    for i in range(n_runs):
        out_path = tmp_dir / f"rust_embed_{i}.safetensors"
        exp_path = None
        if i == export_inputs_on_run:
            exp_path = tmp_dir / "rust_encoder_inputs.safetensors"

        print(f"  Run {i+1}/{n_runs} …", end="", flush=True)
        t, wall = run_rust_embed(weights_path, config_path, fif_path,
                                  out_path, exp_path, device=device,
                                  embed_bin=embed_bin)
        if t is None:
            print(f"  FAILED  (wall={wall:.0f}ms)")
            continue
        print(f"  ✓  {t['total_ms']:.0f} ms  "
              f"(weights={t['weights_ms']:.0f}ms  "
              f"preproc={t['preproc_ms']:.0f}ms  "
              f"encode={t['encode_ms']:.0f}ms)")
        timings.append(t)
        if exp_path and exp_path.exists():
            inputs_path = exp_path

    return timings, inputs_path


# ── Python encoder comparison ─────────────────────────────────────────────────

def load_safetensors_numpy(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Load all tensors from a safetensors file as float32 numpy arrays."""
    result: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            result[key] = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    return result


def run_python_encoder_comparison(
    weights_path: pathlib.Path,
    config:       dict,
    inputs_path:  pathlib.Path,
    rust_embeddings_path: pathlib.Path,
) -> Optional[dict]:
    """
    Load pre-tokenised encoder inputs from `inputs_path`, run the NumPy encoder,
    compare with Rust embeddings from `rust_embeddings_path`.

    Returns a dict of precision metrics.
    """
    if not HAS_ST:
        print("  ⚠ safetensors not available — skipping Python encoder comparison")
        return None

    print("\n▶ Loading weights for Python NumPy encoder …")
    t0 = time.perf_counter()
    weights = load_weights_f32(weights_path)
    ms_load = (time.perf_counter() - t0) * 1000
    print(f"  Loaded in {ms_load:.0f} ms")

    encoder = ZunaEncoderNumpy(weights, config)

    # Load encoder inputs (exported by Rust --export-inputs)
    inputs = load_safetensors_numpy(inputs_path)
    n_epochs = int(inputs.get("n_epochs", np.array([0.0]))[0])
    if n_epochs == 0:
        print("  ⚠ No epochs in inputs file")
        return None

    # Load Rust embeddings for comparison
    rust_embs = load_safetensors_numpy(rust_embeddings_path)

    metrics_per_epoch = []
    py_times = []
    all_py_flat:   List[np.ndarray] = []
    all_rust_flat: List[np.ndarray] = []

    print(f"  Running NumPy encoder on {n_epochs} epoch(s) …")
    for i in range(n_epochs):
        enc_input_key = f"encoder_input_{i}"
        tok_idx_key   = f"tok_idx_{i}"
        emb_key       = f"embeddings_{i}"

        if enc_input_key not in inputs or tok_idx_key not in inputs:
            print(f"  ⚠ Missing epoch {i} in inputs file")
            continue
        if emb_key not in rust_embs:
            print(f"  ⚠ Missing epoch {i} in Rust embeddings")
            continue

        enc_input = inputs[enc_input_key][np.newaxis]             # [1, S, 32]
        tok_idx   = inputs[tok_idx_key].astype(np.int64)          # [S, 4]
        rust_emb  = rust_embs[emb_key]                            # [S, 32]

        t0    = time.perf_counter()
        py_emb = encoder.forward(enc_input, tok_idx)               # [S, 32]
        ms_py  = (time.perf_counter() - t0) * 1000
        py_times.append(ms_py)

        # Compute precision metrics
        diff     = py_emb.astype(np.float64) - rust_emb.astype(np.float64)
        mae      = np.mean(np.abs(diff))
        rmse     = np.sqrt(np.mean(diff ** 2))
        max_err  = np.max(np.abs(diff))
        py_flat  = py_emb.flatten()
        rs_flat  = rust_emb.flatten()
        corr     = float(np.corrcoef(py_flat, rs_flat)[0, 1])
        rel_err  = mae / (np.mean(np.abs(rust_emb)) + 1e-8)

        m = {
            "epoch":   i,
            "n_tokens": py_emb.shape[0],
            "n_dims":   py_emb.shape[1],
            "mae":      float(mae),
            "rmse":     float(rmse),
            "max_err":  float(max_err),
            "pearson_r": corr,
            "rel_err":  float(rel_err),
            "py_ms":    ms_py,
        }
        metrics_per_epoch.append(m)
        all_py_flat.append(py_emb.flatten().astype(np.float32))
        all_rust_flat.append(rust_emb.flatten().astype(np.float32))

        print(f"  Epoch {i}: MAE={mae:.2e}  RMSE={rmse:.2e}  "
              f"maxErr={max_err:.2e}  r={corr:.6f}  py={ms_py:.0f}ms")

    if not metrics_per_epoch:
        return None

    avg = lambda key: float(np.mean([m[key] for m in metrics_per_epoch]))
    return {
        "ms_weights_load_py": ms_load,
        "ms_encode_py_per_epoch": float(np.mean(py_times)),
        "epochs": metrics_per_epoch,
        "summary": {
            "mae":       avg("mae"),
            "rmse":      avg("rmse"),
            "max_err":   avg("max_err"),
            "pearson_r": avg("pearson_r"),
            "rel_err":   avg("rel_err"),
        },
        "all_py":   np.concatenate(all_py_flat) if all_py_flat else np.array([]),
        "all_rust": np.concatenate(all_rust_flat) if all_rust_flat else np.array([]),
    }


# ── Chart generators ──────────────────────────────────────────────────────────

def _ensure_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _add_platform_footer(fig: "plt.Figure", info: dict) -> None:
    """
    Stamp a compact platform-info footer at the bottom of every saved figure.
    Calls subplots_adjust(bottom=0.10) to prevent overlap with axis labels.
    """
    _p: List[str] = [
        f"{info['os']} {info['os_release']}",
        info["arch"],
        info["device_type"].upper(),
        info["cpu_model"],
    ]
    _pc = info.get("cpu_cores_physical")
    _lc = info.get("cpu_cores_logical", 1)
    if _pc and _lc and _pc != _lc:
        _p.append(f"{_pc}P/{_lc}L cores")
    elif _lc:
        _p.append(f"{_lc} cores")
    if info.get("ram_gb"):
        _p.append(f"{info['ram_gb']} GB RAM")
    if info.get("device_type") == "gpu" and info.get("gpu_model"):
        _p.append(f"GPU: {info['gpu_model']}")
        if info.get("gpu_vram_gb"):
            _p.append(f"{info['gpu_vram_gb']} GB VRAM")
    if info.get("estimated_tflops"):
        _p.append(f"~{info['estimated_tflops']} TFLOPS (fp32)")
    footer = "  ·  ".join(_p)
    fig.subplots_adjust(bottom=0.10)
    fig.text(
        0.5, 0.02, footer,
        ha="center", va="bottom", fontsize=7, color="#555555",
        style="italic", transform=fig.transFigure, clip_on=False,
    )


def save_speed_chart(
    timings: List[dict],
    backend_name: str,
    platform_info: dict,
) -> pathlib.Path:
    """Grouped bar chart: per-phase timing across N runs."""
    path = FIGURES_DIR / f"bench_speed_{platform_info['slug']}.png"
    if not HAS_MPL or not timings:
        return path
    _ensure_figures()

    phases   = ["weights_ms", "preproc_ms", "encode_ms"]
    labels   = ["Weight load", "Preprocess", "Encode"]
    n_runs   = len(timings)
    x        = np.arange(len(phases))
    width    = 0.6 / max(n_runs, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: grouped bars per run
    ax = axes[0]
    for ri, t in enumerate(timings):
        vals = [t.get(ph, 0) for ph in phases]
        ax.bar(x + ri * width - width * n_runs / 2, vals, width,
               label=f"Run {ri+1}", color=PALETTE[ri % len(PALETTE)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(f"Per-phase timing — {n_runs} runs\n({backend_name})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: mean ± std stacked bar
    ax = axes[1]
    means = [np.mean([t.get(ph, 0) for t in timings]) for ph in phases]
    stds  = [np.std([t.get(ph, 0)  for t in timings]) for ph in phases]
    bars  = ax.bar(labels, means, color=PALETTE[:3], alpha=0.85, zorder=2)
    ax.errorbar(labels, means, yerr=stds, fmt="none", capsize=6,
                ecolor="black", linewidth=2, zorder=3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 3,
                f"{m:.0f}±{s:.0f}ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(f"Mean ± std across {n_runs} runs", fontsize=12)
    ax.grid(axis="y", alpha=0.3, zorder=1)

    fig.suptitle("ZUNA Rust Encoder — Speed Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


def save_precision_chart(cmp: dict, platform_info: dict) -> pathlib.Path:
    """Bar chart of Python vs Rust precision metrics."""
    path = FIGURES_DIR / f"bench_precision_{platform_info['slug']}.png"
    if not HAS_MPL or not cmp:
        return path
    _ensure_figures()

    s = cmp["summary"]
    metrics = {
        "MAE":       s["mae"],
        "RMSE":      s["rmse"],
        "Max Error": s["max_err"],
        "Rel Error": s["rel_err"],
    }
    r = s["pearson_r"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: error bars
    ax = axes[0]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=PALETTE[:4], alpha=0.85)
    for bar, v in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Error (float32 units)", fontsize=11)
    ax.set_title("Precision: NumPy vs Rust encoder\n(same tokenised input)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # Right: scatter plot
    ax = axes[1]
    py   = cmp.get("all_py",   np.array([]))
    rust = cmp.get("all_rust", np.array([]))
    if len(py) > 0:
        # Subsample for plotting
        idx = np.random.choice(len(py), size=min(5000, len(py)), replace=False)
        ax.scatter(rust[idx], py[idx], s=1, alpha=0.25, color=PALETTE[0])
        lim = max(np.abs(rust[idx]).max(), np.abs(py[idx]).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.2, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel("Rust encoder output", fontsize=11)
    ax.set_ylabel("Python NumPy encoder output", fontsize=11)
    ax.set_title(f"Value scatter  (Pearson r = {r:.6f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    fig.suptitle("Python NumPy vs Rust — Encoder Precision", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


def save_distribution_chart(
    rust_embs_all: np.ndarray,
    py_embs_all:   Optional[np.ndarray],
    platform_info: dict,
) -> pathlib.Path:
    """Histogram of embedding values vs N(0,1); overlay Python if available."""
    path = FIGURES_DIR / f"bench_distribution_{platform_info['slug']}.png"
    if not HAS_MPL:
        return path
    _ensure_figures()

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-5, 5, 80)

    ax.hist(rust_embs_all, bins=bins, density=True, alpha=0.65,
            color=PALETTE[0], label="Rust embeddings")
    if py_embs_all is not None and len(py_embs_all) > 0:
        ax.hist(py_embs_all, bins=bins, density=True, alpha=0.45,
                color=PALETTE[1], label="Python (NumPy) embeddings")

    # N(0,1) reference
    xs = np.linspace(-5, 5, 300)
    ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            "r--", linewidth=2, label="N(0,1) ideal")

    ax.set_xlabel("Embedding value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Embedding distribution (MMD regularlisation → N(0,1))", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # KL divergence annotation
    if HAS_SCIPY and len(rust_embs_all) > 100:
        counts, edges = np.histogram(rust_embs_all, bins=80, density=True)
        ref           = np.exp(-0.5 * ((edges[:-1] + edges[1:]) / 2) ** 2) / np.sqrt(2 * np.pi)
        counts        = np.maximum(counts, 1e-12)
        ref           = np.maximum(ref,    1e-12)
        kl = float(np.sum(rel_entr(counts / counts.sum(), ref / ref.sum())))
        ax.text(0.97, 0.95, f"KL(Rust ∥ N(0,1)) = {kl:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


def save_dim_stats_chart(
    rust_embs_all: np.ndarray,
    n_dims: int,
    py_embs_all:   Optional[np.ndarray] = None,
    platform_info: dict = None,
) -> pathlib.Path:
    """Per-dimension mean ± std for Rust (and Python if available)."""
    path = FIGURES_DIR / f"bench_dim_stats_{platform_info['slug']}.png"
    if not HAS_MPL or n_dims == 0:
        return path
    _ensure_figures()

    def dim_stats(flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mat    = flat[: (len(flat) // n_dims) * n_dims].reshape(-1, n_dims)
        return mat.mean(axis=0), mat.std(axis=0)

    r_mean, r_std = dim_stats(rust_embs_all)
    x = np.arange(n_dims)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: means
    ax = axes[0]
    ax.bar(x, r_mean, color=PALETTE[0], alpha=0.8, label="Rust mean")
    if py_embs_all is not None and len(py_embs_all) >= n_dims:
        p_mean, _ = dim_stats(py_embs_all)
        ax.bar(x + 0.4, p_mean, color=PALETTE[1], alpha=0.6, width=0.4, label="Python mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean", fontsize=11)
    ax.set_title("Per-latent-dimension statistics  (ideal mean≈0, std≈1)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: stds
    ax = axes[1]
    ax.bar(x, r_std, color=PALETTE[2], alpha=0.8, label="Rust std")
    if py_embs_all is not None and len(py_embs_all) >= n_dims:
        _, p_std = dim_stats(py_embs_all)
        ax.bar(x + 0.4, p_std, color=PALETTE[3], alpha=0.6, width=0.4, label="Python std")
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", label="Ideal std=1")
    ax.set_xlabel("Latent dimension", fontsize=11)
    ax.set_ylabel("Std", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


def save_run_consistency_chart(timings: List[dict], platform_info: dict) -> pathlib.Path:
    """Box-plot style chart showing variance across N Rust runs."""
    path = FIGURES_DIR / f"bench_run_consistency_{platform_info['slug']}.png"
    if not HAS_MPL or len(timings) < 2:
        return path
    _ensure_figures()

    phases = ["weights_ms", "preproc_ms", "encode_ms", "total_ms"]
    labels = ["Weight load", "Preprocess", "Encode", "Total"]
    data   = [[t.get(ph, 0) for t in timings] for ph in phases]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual points
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, color="black", s=30, zorder=5, alpha=0.7)

    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(f"Run-to-run consistency ({len(timings)} Rust runs)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


def save_py_vs_rust_error_chart(cmp: dict, platform_info: dict) -> pathlib.Path:
    """Histogram of per-element error between Python and Rust."""
    path = FIGURES_DIR / f"bench_py_vs_rust_{platform_info['slug']}.png"
    if not HAS_MPL or not cmp:
        return path
    _ensure_figures()

    py   = cmp.get("all_py",   np.array([]))
    rust = cmp.get("all_rust", np.array([]))
    if len(py) == 0:
        return path

    diff = (py - rust).astype(np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Error histogram
    ax = axes[0]
    ax.hist(diff, bins=80, color=PALETTE[0], alpha=0.8, density=True)
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Python − Rust (f32 units)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Element-wise error distribution", fontsize=11)
    mu, sigma = diff.mean(), diff.std()
    ax.text(0.97, 0.95,
            f"μ = {mu:.2e}\nσ = {sigma:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Cumulative absolute error
    ax = axes[1]
    sorted_abs = np.sort(np.abs(diff))
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax.semilogx(sorted_abs, cdf, color=PALETTE[0], linewidth=1.8)
    ax.axvline(1e-3, color="grey", linestyle="--", linewidth=1, label="|err|=1e-3")
    ax.axvline(1e-4, color="grey", linestyle=":",  linewidth=1, label="|err|=1e-4")
    pct_below_1e3 = float(np.mean(sorted_abs < 1e-3) * 100)
    ax.text(0.97, 0.15, f"{pct_below_1e3:.1f}% < 1e-3",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("|Python − Rust|  (log scale)", fontsize=11)
    ax.set_ylabel("Cumulative fraction", fontsize=11)
    ax.set_title("Cumulative absolute error", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle("Python NumPy vs Rust — Element-wise Error Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _add_platform_footer(fig, platform_info)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart → {path}")
    return path


# ── README updater ────────────────────────────────────────────────────────────

_BENCH_MARKER_START = "<!-- BENCHMARK_START -->"
_BENCH_MARKER_END   = "<!-- BENCHMARK_END -->"

def _fmt_ms(ms: float) -> str:
    return f"{ms:.1f} ms" if ms < 1000 else f"{ms/1000:.2f} s"


def update_readme(
    timings:       List[dict],
    cmp:           Optional[dict],
    rust_embs:     Optional[np.ndarray],
    n_dims:        int,
    platform_info: dict,
) -> None:
    """Insert/replace the benchmark section in README.md."""
    if not README_PATH.exists():
        return

    avg  = lambda key: float(np.mean([t.get(key, 0) for t in timings])) if timings else 0.0
    std  = lambda key: float(np.std( [t.get(key, 0) for t in timings])) if timings else 0.0
    slug = platform_info["slug"]

    lines = [
        _BENCH_MARKER_START,
        "## 📊 Benchmark results",
        "",
        "> Auto-generated by `bench_and_visualize.py` — do not edit manually.",
        "",
        f"**Platform**: {platform_info['label']} · `{len(timings)}` runs",
        "",
        "### Speed",
        "",
        "| Phase          | Mean (ms) | Std (ms) |",
        "|:---------------|----------:|---------:|",
        f"| Weight loading | {avg('weights_ms'):>9.1f} | {std('weights_ms'):>8.1f} |",
        f"| Preprocess FIF | {avg('preproc_ms'):>9.1f} | {std('preproc_ms'):>8.1f} |",
        f"| Encoder fwd    | {avg('encode_ms'):>9.1f} | {std('encode_ms'):>8.1f} |",
        f"| **Total**      | **{avg('total_ms'):>6.1f}** | {std('total_ms'):>8.1f} |",
        "",
        f"![Speed](./figures/bench_speed_{slug}.png)",
        "",
    ]

    if cmp:
        s = cmp["summary"]
        lines += [
            "### Python NumPy vs Rust precision",
            "",
            "Both implementations receive identical pre-tokenised EEG tensors;",
            "differences reflect float32 rounding order only.",
            "",
            "| Metric          | Value |",
            "|:----------------|------:|",
            f"| MAE             | `{s['mae']:.2e}` |",
            f"| RMSE            | `{s['rmse']:.2e}` |",
            f"| Max abs error   | `{s['max_err']:.2e}` |",
            f"| Pearson r       | `{s['pearson_r']:.6f}` |",
            f"| Relative error  | `{s['rel_err']:.2e}` |",
            f"| Python encode   | `{cmp['ms_encode_py_per_epoch']:.1f} ms/epoch` |",
            "",
            f"![Precision](./figures/bench_precision_{slug}.png)",
            f"![Error](./figures/bench_py_vs_rust_{slug}.png)",
            "",
        ]

    if rust_embs is not None and len(rust_embs) > 0:
        flat_mean = float(rust_embs.mean())
        flat_std  = float(rust_embs.std())
        lines += [
            "### Embedding distribution (MMD regularlisation)",
            "",
            "ZUNA trains with an MMD loss that pushes embeddings toward **N(0, I)**.",
            "",
            f"| Stat          | Value |",
            f"|:--------------|------:|",
            f"| Global mean   | `{flat_mean:+.4f}` |",
            f"| Global std    | `{flat_std:.4f}` |",
            f"| n_dims        | `{n_dims}` |",
            "",
            f"![Distribution](./figures/bench_distribution_{slug}.png)",
            f"![Dim stats](./figures/bench_dim_stats_{slug}.png)",
            "",
        ]

    lines += [
        "### Run consistency",
        "",
        f"![Consistency](./figures/bench_run_consistency_{slug}.png)",
        "",
        _BENCH_MARKER_END,
    ]

    text = README_PATH.read_text()
    bench_block = "\n".join(lines)

    if _BENCH_MARKER_START in text and _BENCH_MARKER_END in text:
        before = text[: text.index(_BENCH_MARKER_START)]
        after  = text[text.index(_BENCH_MARKER_END) + len(_BENCH_MARKER_END):]
        new_text = before + bench_block + after
    else:
        new_text = text.rstrip() + "\n\n" + bench_block + "\n"

    README_PATH.write_text(new_text)
    print(f"  README.md updated with benchmark results.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo",      default=REPO_ID,  help="HuggingFace repo ID")
    ap.add_argument("--weights",   default=None,     help="Explicit safetensors weights path")
    ap.add_argument("--config",    default=None,     help="Explicit config.json path")
    ap.add_argument("--fif",       default=str(SAMPLE_FIF), help="Input FIF file")
    ap.add_argument("--runs",      type=int, default=3, help="Number of Rust timing runs")
    ap.add_argument("--device",    default="cpu",    help="Compute device passed to Rust binary (cpu / gpu)")
    ap.add_argument("--embed-bin", default=None,     help="Path to pre-built embed binary (skips cargo build)")
    ap.add_argument("--no-python-encoder", action="store_true",
                    help="Skip Python NumPy encoder comparison")
    ap.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    ap.add_argument("--no-readme", action="store_true", help="Skip README update")
    args = ap.parse_args()

    print("=" * 64)
    print("  ZUNA bench_and_visualize.py")
    print("=" * 64)

    # ── Collect platform info ──────────────────────────────────────────────────
    print("\n▶ Detecting platform …")
    platform_info = collect_platform_info(args.device)
    print(f"  {platform_info['label']}")
    print(f"  slug : {platform_info['slug']}")

    # ── Resolve weights ────────────────────────────────────────────────────────
    print("\n▶ Resolving model weights …")
    weights_path, config_path = resolve_weights(
        args.repo, args.weights, args.config
    )
    print(f"  weights : {weights_path}")
    print(f"  config  : {config_path}")
    config = load_config(config_path)

    # ── Rust timing benchmark ──────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="zuna_bench_") as tmp_str:
        tmp_dir = pathlib.Path(tmp_str)

        embed_bin = pathlib.Path(args.embed_bin) if args.embed_bin else None
        timings, inputs_path = run_rust_benchmark(
            weights_path         = weights_path,
            config_path          = config_path,
            fif_path             = pathlib.Path(args.fif),
            n_runs               = args.runs,
            tmp_dir              = tmp_dir,
            export_inputs_on_run = 0,   # always export on first run
            device               = args.device,
            embed_bin            = embed_bin,
        )

        if not timings:
            print("⚠  All Rust runs failed. Check that `cargo run --example embed --release` works.")
            sys.exit(1)

        # Load Rust embeddings from run 0 for distribution analysis
        rust_embs_path = tmp_dir / "rust_embed_0.safetensors"
        rust_embs_all:  Optional[np.ndarray] = None
        n_dims = 0
        if HAS_ST and rust_embs_path.exists():
            embs_dict = load_safetensors_numpy(rust_embs_path)
            chunks    = [embs_dict[k] for k in sorted(embs_dict) if k.startswith("embeddings_")]
            if chunks:
                rust_embs_all = np.concatenate([c.flatten() for c in chunks])
                n_dims = chunks[0].shape[-1] if chunks else 0
                print(f"\n  Rust embeddings: {len(chunks)} epoch(s)  "
                      f"n_dims={n_dims}  total_values={len(rust_embs_all)}")
                print(f"  mean={rust_embs_all.mean():+.4f}  "
                      f"std={rust_embs_all.std():.4f}  "
                      f"min={rust_embs_all.min():.3f}  max={rust_embs_all.max():.3f}")

        # ── Python encoder comparison ──────────────────────────────────────────
        cmp: Optional[dict] = None
        py_embs_all: Optional[np.ndarray] = None
        if (not args.no_python_encoder and HAS_ST
                and inputs_path and inputs_path.exists()):
            cmp = run_python_encoder_comparison(
                weights_path         = weights_path,
                config               = config,
                inputs_path          = inputs_path,
                rust_embeddings_path = rust_embs_path,
            )
            if cmp:
                py_embs_all = cmp.get("all_py")
        else:
            if not HAS_ST:
                print("\n⚠  Skipping Python encoder comparison (safetensors not installed)")
            elif args.no_python_encoder:
                print("\n  Skipping Python encoder comparison (--no-python-encoder)")
            elif not inputs_path:
                print("\n⚠  Skipping Python encoder comparison (Rust failed to export inputs)")

        # ── Print summary ──────────────────────────────────────────────────────
        print("\n" + "=" * 64)
        print("  RESULTS SUMMARY")
        print("=" * 64)
        _avg = lambda k: float(np.mean([t.get(k, 0) for t in timings]))
        _std = lambda k: float(np.std( [t.get(k, 0) for t in timings]))
        print(f"  Platform : {platform_info['label']}")
        print(f"  Rust encoder ({len(timings)} runs):")
        for phase, lbl in [("weights_ms","Weights"), ("preproc_ms","Preprocess"),
                            ("encode_ms","Encode"), ("total_ms","Total")]:
            print(f"    {lbl:12s}  {_avg(phase):7.1f} ± {_std(phase):.1f} ms")
        if cmp:
            s = cmp["summary"]
            print(f"\n  Python vs Rust precision:")
            print(f"    MAE        {s['mae']:.2e}")
            print(f"    RMSE       {s['rmse']:.2e}")
            print(f"    Max error  {s['max_err']:.2e}")
            print(f"    Pearson r  {s['pearson_r']:.6f}")
        if rust_embs_all is not None:
            print(f"\n  Embedding distribution (Rust):")
            print(f"    mean={rust_embs_all.mean():+.4f}  std={rust_embs_all.std():.4f}  "
                  f"(ideal: 0.0 and 1.0)")

        # ── Save raw benchmark data ────────────────────────────────────────────
        _ensure_figures()
        slug = platform_info["slug"]
        bench_data_path = FIGURES_DIR / f"bench_data_{slug}.json"
        bench_data = {
            "platform": {k: v for k, v in platform_info.items() if k != "label"},
            "rust_timings": timings,
            "n_runs":       args.runs,
            "python_comparison": {k: v for k, v in (cmp or {}).items()
                                  if not isinstance(v, np.ndarray)},
            "embedding_stats": {
                "mean":   float(rust_embs_all.mean()) if rust_embs_all is not None else None,
                "std":    float(rust_embs_all.std())  if rust_embs_all is not None else None,
                "n_dims": n_dims,
            },
        }
        with open(bench_data_path, "w") as f:
            json.dump(bench_data, f, indent=2, default=str)
        print(f"\n  Data → {bench_data_path}")

        # ── Charts ─────────────────────────────────────────────────────────────
        if not args.no_charts and HAS_MPL:
            print("\n▶ Generating charts …")
            backend_info = f"{args.device.upper()} (NdArray + {'Rayon' if args.device == 'cpu' else 'wgpu'})"
            save_speed_chart(timings, backend_info, platform_info)
            save_run_consistency_chart(timings, platform_info)
            if rust_embs_all is not None:
                save_distribution_chart(rust_embs_all, py_embs_all, platform_info)
                save_dim_stats_chart(rust_embs_all, n_dims, py_embs_all, platform_info)
            if cmp:
                save_precision_chart(cmp, platform_info)
                save_py_vs_rust_error_chart(cmp, platform_info)
        elif not HAS_MPL:
            print("\n  ⚠ Charts skipped (matplotlib not installed)")

        # ── Update README ──────────────────────────────────────────────────────
        if not args.no_readme:
            print("\n▶ Updating README.md …")
            update_readme(timings, cmp, rust_embs_all, n_dims, platform_info)

    print("\n✓ bench_and_visualize.py complete.")
    print(f"  Charts  : {FIGURES_DIR}/")
    print(f"  Data    : {bench_data_path}")


if __name__ == "__main__":
    main()
