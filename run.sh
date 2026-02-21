#!/bin/sh
# run.sh — ZUNA EEG Foundation Model — end-to-end pure-Rust inference
#
# Usage (from repo root):
#   sh zuna-rs/run.sh                        # build + run (GPU on macOS, CPU on Linux)
#   sh zuna-rs/run.sh --cached               # skip build if binary already exists
#   MODE=cpu sh zuna-rs/run.sh               # force CPU
#   MODE=both sh zuna-rs/run.sh              # run both, compare timing + precision
#   STEPS=10 sh zuna-rs/run.sh              # faster (fewer diffusion steps)
#   VERBOSE=1 sh zuna-rs/run.sh             # electrode table + per-epoch stats
#   FIF_FILE=other.fif sh zuna-rs/run.sh   # different recording

set -e

# ── Parse flags ────────────────────────────────────────────────────────────────
CACHED=0
for arg in "$@"; do
    case "$arg" in
        --cached) CACHED=1 ;;
        *) printf 'Unknown arg: %s\n' "$arg" >&2; exit 1 ;;
    esac
done

# ── Paths (all relative to repo root) ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

relpath() { r="${1#$REPO_ROOT/}"; printf '%s' "${r:-$1}"; }

FIF_FILE="${FIF_FILE:-$SCRIPT_DIR/data/sample1_raw.fif}"
WORKING_DIR="$REPO_ROOT/working"

STEPS="${STEPS:-50}"
CFG="${CFG:-1.0}"
DATA_NORM="${DATA_NORM:-10.0}"
VERBOSE="${VERBOSE:-0}"
BENCH_RUNS="${BENCH_RUNS:-1}"

# Binaries in /tmp (host-shared mount may be noexec)
CPU_BIN=/tmp/zuna-infer-cpu
GPU_BIN=/tmp/zuna-infer-gpu
DOWNLOAD_BIN=/tmp/zuna-download-weights

CPU_TARGET=/tmp/zuna-rs-target
GPU_TARGET=/tmp/zuna-rs-gpu-target

OUTPUT_CPU="$WORKING_DIR/output_cpu.safetensors"
OUTPUT_GPU="$WORKING_DIR/output_gpu.safetensors"
OUTPUT_FILE="$WORKING_DIR/output.safetensors"

# ── Platform + default MODE ────────────────────────────────────────────────────
OS="$(uname -s)"
if [ -z "$MODE" ]; then
    [ "$OS" = "Darwin" ] && MODE=gpu || MODE=cpu
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
step() { printf '\n\033[34m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m✓ %s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '\033[33m⚠  %s\033[0m\n' "$*"; }

has_vulkan() { [ -e /dev/dri/renderD128 ] || command -v vulkaninfo >/dev/null 2>&1; }

# ── Thread counts (export so BLAS and Rayon pick them up) ─────────────────────
NCPUS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
export RAYON_NUM_THREADS="$NCPUS"
export VECLIB_MAXIMUM_THREADS="$NCPUS"   # Apple Accelerate
export OMP_NUM_THREADS="$NCPUS"          # OpenBLAS / OpenMP

# ── Sanity checks ─────────────────────────────────────────────────────────────
step "Environment"
[ -f "$FIF_FILE" ] || die "FIF not found: $(relpath "$FIF_FILE")"
case "$MODE" in cpu|gpu|both) ;;
    *) die "MODE must be cpu | gpu | both (got: $MODE)";; esac
mkdir -p "$WORKING_DIR"
ok "FIF    : $(relpath "$FIF_FILE")"
ok "Mode   : $MODE  (steps=$STEPS  bench_runs=$BENCH_RUNS  cached=$CACHED)"
ok "Threads: $NCPUS"

# ── Step 1: Build ──────────────────────────────────────────────────────────────
step "[1/3] Build"
. "$HOME/.cargo/env" 2>/dev/null || true

# build_bin FEATURES TARGET_DIR OUT_BIN LABEL
build_bin() {
    _features="$1"; _target="$2"; _out="$3"; _label="$4"

    if [ "$CACHED" = "1" ] && [ -f "$_out" ]; then
        ok "$_label: using cached  $_out"
        return 0
    fi

    info "Building $_label binary  (cargo build --release $_features)"
    info "(Rust incremental compile — only changed crates are recompiled)"
    (cd "$SCRIPT_DIR" && \
        CARGO_TARGET_DIR="$_target" \
        cargo build --release $_features --bin infer) || die "$_label build failed"
    cp "$_target/release/infer" "$_out"
    chmod +x "$_out"
    ok "$_label: $_out"
}

# ── CPU binary ────────────────────────────────────────────────────────────────
if [ "$MODE" = "cpu" ] || [ "$MODE" = "both" ]; then
    if [ "$OS" = "Darwin" ]; then
        CPU_FEATURES="--features blas-accelerate,hf-download"
    elif [ -f /etc/alpine-release ]; then
        # Alpine Linux (musl): openblas-src build script needs static OpenSSL,
        # which is unavailable on musl.  Pure Rayon + SIMD is fast enough.
        CPU_FEATURES=""
    elif ldconfig -p 2>/dev/null | grep -q libopenblas; then
        CPU_FEATURES="--features openblas-system"
    else
        CPU_FEATURES=""
    fi
    build_bin "$CPU_FEATURES" "$CPU_TARGET" "$CPU_BIN" "CPU"
fi

# ── GPU binary ────────────────────────────────────────────────────────────────
if [ "$MODE" = "gpu" ] || [ "$MODE" = "both" ]; then
    if [ "$OS" != "Darwin" ] && ! has_vulkan; then
        warn "No Vulkan GPU on Linux — falling back to CPU"
        MODE=cpu
    else
        [ "$OS" = "Darwin" ] \
            && _gpu_desc="wgpu + Metal (macOS)" \
            || _gpu_desc="wgpu + Vulkan (Linux)"
        build_bin "--no-default-features --features wgpu" \
                  "$GPU_TARGET" "$GPU_BIN" "GPU ($_gpu_desc)"
    fi
fi

# ── download_weights (macOS only) ─────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
    if [ "$CACHED" = "1" ] && [ -f "$DOWNLOAD_BIN" ]; then
        ok "download_weights: cached"
    else
        info "Building download_weights …"
        (cd "$SCRIPT_DIR" && \
            CARGO_TARGET_DIR="$CPU_TARGET" \
            cargo build --release --features blas-accelerate,hf-download \
                --bin download_weights) && \
        cp "$CPU_TARGET/release/download_weights" "$DOWNLOAD_BIN" && \
        chmod +x "$DOWNLOAD_BIN" || warn "download_weights build failed (non-fatal)"
    fi
fi

# ── Step 2: Weights ────────────────────────────────────────────────────────────
step "[2/3] Weights"
PATHS=""

[ -f "$DOWNLOAD_BIN" ] && PATHS=$("$DOWNLOAD_BIN" 2>/dev/null) || true

if [ -z "$PATHS" ]; then
    # Find Python — check $ZUNA_PYTHON (set by benchmark.sh), then python3, then python
    _py="${ZUNA_PYTHON:-}"
    if [ -z "$_py" ]; then
        for _cand in "${VENV_PYTHON:-}" python3 python; do
            [ -z "$_cand" ] && continue
            command -v "$_cand" >/dev/null 2>&1 \
                && "$_cand" -c "import sys; sys.exit(0)" 2>/dev/null \
                && { _py="$_cand"; break; }
        done
    fi

    if [ -n "$_py" ]; then
        # Use ZUNA_W: / ZUNA_C: prefixed output so conda activation noise
        # ("Requirement already satisfied: numpy") can't corrupt path parsing.
        _raw="$("$_py" - 2>/dev/null <<'PYEOF'
import sys, pathlib, os

def emit(w, c):
    sys.stdout.write(f"ZUNA_W:{w}\nZUNA_C:{c}\n")
    sys.stdout.flush()
    sys.exit(0)

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for p in pathlib.Path(hf_home).rglob("model-00001-of-00001.safetensors"):
    c = p.parent / "config.json"
    if c.exists():
        emit(p, c)

try:
    from huggingface_hub import snapshot_download
    snap = pathlib.Path(snapshot_download("Zyphra/ZUNA"))
    emit(snap / "model-00001-of-00001.safetensors", snap / "config.json")
except Exception as e:
    sys.stderr.write(f"HF: {e}\n")
sys.exit(1)
PYEOF
        )" || true
        _w=$(printf '%s\n' "$_raw" | grep '^ZUNA_W:' | head -1 | sed 's/^ZUNA_W://')
        _c=$(printf '%s\n' "$_raw" | grep '^ZUNA_C:' | head -1 | sed 's/^ZUNA_C://')
        [ -n "$_w" ] && [ -n "$_c" ] && PATHS="$_w
$_c"
    fi
fi

if [ -z "$PATHS" ]; then
    _w=$(find "$HOME/.cache/huggingface" -name "model-00001-of-00001.safetensors" 2>/dev/null | head -1)
    _c=$(find "$HOME/.cache/huggingface" -name "config.json" -path "*/Zyphra*ZUNA*" 2>/dev/null | head -1)
    [ -n "$_w" ] && [ -n "$_c" ] && PATHS="$_w
$_c"
fi

_py_hint="${_py:-python3}"
[ -n "$PATHS" ] || die \
    "Cannot find weights. Run: $_py_hint -c \"from huggingface_hub import \
snapshot_download; snapshot_download('Zyphra/ZUNA')\""

WEIGHTS_FILE=$(printf '%s\n' "$PATHS" | head -1)
CONFIG_FILE=$(printf  '%s\n' "$PATHS" | sed -n '2p')
[ -f "$WEIGHTS_FILE" ] || die "Weights missing: $WEIGHTS_FILE"
[ -f "$CONFIG_FILE"  ] || die "Config missing:  $CONFIG_FILE"
ok "Weights: $WEIGHTS_FILE"
ok "Config : $CONFIG_FILE"

# ── Step 3: Inference ──────────────────────────────────────────────────────────
step "[3/3] Inference  (mode=$MODE  steps=$STEPS  cfg=$CFG)"

VERBOSE_FLAG=""; [ "$VERBOSE" = "1" ] && VERBOSE_FLAG="--verbose"

# run_mode BIN OUTPUT LABEL
run_mode() {
    _bin="$1"; _out="$2"; _label="$3"
    info "━━━ $_label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _best=2147483647; _i=0
    while [ $_i -lt "$BENCH_RUNS" ]; do
        _i=$((_i+1))
        _t0=$(date +%s%N 2>/dev/null || echo 0)
        "$_bin" \
            --weights "$WEIGHTS_FILE" --config "$CONFIG_FILE" \
            --fif "$FIF_FILE" --output "$_out" \
            --steps "$STEPS" --cfg "$CFG" --data-norm "$DATA_NORM" \
            $VERBOSE_FLAG 2>&1
        _t1=$(date +%s%N 2>/dev/null || echo 0)
        _ms=$(( (_t1-_t0)/1000000 ))
        [ $_ms -lt $_best ] && _best=$_ms
        [ "$BENCH_RUNS" -gt 1 ] && info "  run $_i/$BENCH_RUNS: ${_ms} ms"
    done
    printf '\n  \033[32m%s best: %d ms  (%.1f s)\033[0m\n\n' \
        "$_label" "$_best" "$(echo "$_best" | awk '{printf "%.1f",$1/1000}')"
    eval "TIME_${_label}=$_best"
}

[ "$MODE" = "cpu" ] || [ "$MODE" = "both" ] && run_mode "$CPU_BIN" "$OUTPUT_CPU" "CPU"
[ "$MODE" = "gpu" ] || [ "$MODE" = "both" ] && run_mode "$GPU_BIN" "$OUTPUT_GPU" "GPU"

# Canonical output = best available
if   [ -f "$OUTPUT_GPU" ]; then cp "$OUTPUT_GPU" "$OUTPUT_FILE"
elif [ -f "$OUTPUT_CPU" ]; then cp "$OUTPUT_CPU" "$OUTPUT_FILE"; fi

# ── CPU vs GPU comparison ──────────────────────────────────────────────────────
if [ "$MODE" = "both" ] && [ -f "$OUTPUT_CPU" ] && [ -f "$OUTPUT_GPU" ]; then
    step "CPU vs GPU"
    _sp=$(echo "$TIME_CPU $TIME_GPU" | awk '{printf "%.1f",$1/$2}')
    printf '  %-16s %10s %10s %10s\n' "" "CPU" "GPU" "Speedup"
    printf '  %-16s %9dms %9dms %8s×\n' "Wall-clock" "$TIME_CPU" "$TIME_GPU" "$_sp"
    "${_py:-python3}" - "$OUTPUT_CPU" "$OUTPUT_GPU" 2>/dev/null <<'PYEOF'
import sys,json,numpy as np
from pathlib import Path
def load(p):
    raw=Path(p).read_bytes(); n=int.from_bytes(raw[:8],"little")
    hdr=json.loads(raw[8:8+n]); ds=8+n
    return {k:np.frombuffer(raw[ds+v["data_offsets"][0]:ds+v["data_offsets"][1]],
            dtype="<f4").reshape(v["shape"]) for k,v in hdr.items() if k!="__metadata__"}
cpu,gpu=load(sys.argv[1]),load(sys.argv[2])
keys=sorted(k for k in cpu if k.startswith("reconstructed"))
print(f"  {'Tensor':<22} {'r(cpu,gpu)':>11} {'RMSE':>10} {'Max|Δ|':>12}")
print(f"  {'─'*57}")
for k in keys:
    a,b=cpu[k].ravel().astype("f8"),gpu[k].ravel().astype("f8")
    r=float(np.corrcoef(a,b)[0,1])
    print(f"  {k:<22} {r:>11.6f} {np.sqrt(np.mean((a-b)**2)):>10.3e} {np.abs(a-b).max():>12.3e}")
PYEOF
fi

# ── Done ──────────────────────────────────────────────────────────────────────
step "Done"
ok "Input  : $(relpath "$FIF_FILE")"
ok "Output : $(relpath "$OUTPUT_FILE")"
printf '\n  Inspect  : %s zuna-rs/scripts/inspect_output.py working/output.safetensors\n' "${_py:-python3}"
printf '  Visualise: %s zuna-rs/bench_and_visualize.py\n\n' "${_py:-python3}"
