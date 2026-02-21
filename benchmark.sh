#!/bin/sh
# benchmark.sh — ZUNA encoder benchmark + Python-vs-Rust comparison
#
# Usage:
#   sh zuna-rs/benchmark.sh              # embed timing + bench_and_visualize.py
#   sh zuna-rs/benchmark.sh --full       # also run full encode+decode (infer)
#   sh zuna-rs/benchmark.sh --runs 5     # 5 Rust timing iterations (default 3)
#   sh zuna-rs/benchmark.sh --no-python  # skip NumPy encoder comparison
#   sh zuna-rs/benchmark.sh --cached     # skip build if binary exists
#
# On macOS  → builds with wgpu (Metal GPU).
# On Linux  → Vulkan GPU if /dev/dri present, else CPU (Rayon).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Parse flags ────────────────────────────────────────────────────────────────
CACHED=0; FULL=0; RUNS=3; NO_PYTHON=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cached)    CACHED=1 ;;
        --full)      FULL=1 ;;
        --runs)      shift; RUNS="$1" ;;
        --runs=*)    RUNS="${1#--runs=}" ;;
        --no-python) NO_PYTHON=1 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { printf '\033[31m✗  %s\033[0m\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;34m━━━  %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m  %s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m  %s\n' "$*"; }

# ── Platform ───────────────────────────────────────────────────────────────────
OS="$(uname -s)"
NCPUS="$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
export RAYON_NUM_THREADS="$NCPUS"
export OMP_NUM_THREADS="$NCPUS"
export VECLIB_MAXIMUM_THREADS="$NCPUS"

if [ "$OS" = "Darwin" ]; then
    DEVICE=gpu
    FEATURES="--no-default-features --features wgpu"
    BACKEND_LABEL="wgpu / Metal (macOS GPU)"
elif [ -e /dev/dri/renderD128 ] || command -v vulkaninfo >/dev/null 2>&1; then
    DEVICE=gpu
    FEATURES="--no-default-features --features wgpu"
    BACKEND_LABEL="wgpu / Vulkan (Linux GPU)"
elif [ -f /etc/alpine-release ]; then
    DEVICE=cpu
    FEATURES=""   # Alpine: skip openblas-system (no static OpenSSL on musl)
    BACKEND_LABEL="NdArray + Rayon (Alpine CPU)"
elif ldconfig -p 2>/dev/null | grep -q libopenblas; then
    DEVICE=cpu
    FEATURES="--features openblas-system"
    BACKEND_LABEL="NdArray + OpenBLAS (Linux CPU)"
else
    DEVICE=cpu
    FEATURES=""
    BACKEND_LABEL="NdArray + Rayon (Linux CPU)"
fi

# Allow override
[ -n "$ZUNA_DEVICE" ] && DEVICE="$ZUNA_DEVICE"

step "ZUNA benchmark  —  $BACKEND_LABEL"
info "runs=$RUNS  full=$FULL  cached=$CACHED  threads=$NCPUS"

# ── Font + freetype install (plotters ttf feature needs both) ─────────────────
# On Alpine: freetype-static provides libfreetype.a (needed at link time);
#            ttf-freefont provides font files (needed at chart-render time).
if [ -f /etc/alpine-release ]; then
    _need_pkg=0
    pkg_installed() { apk info -e "$1" >/dev/null 2>&1; }
    pkg_installed ttf-freefont    || _need_pkg=1
    pkg_installed freetype-static || _need_pkg=1
    if [ "$_need_pkg" = "1" ]; then
        info "Installing fonts + freetype (needed for chart rendering) …"
        apk add --quiet ttf-freefont freetype-static 2>/dev/null && fc-cache -f 2>/dev/null \
            || warn "Package install failed — charts may be skipped"
    fi
fi

# ── Find Python ───────────────────────────────────────────────────────────────
# Check explicit venv paths, then 'python3', then 'python'.
# Export as ZUNA_PYTHON so Rust examples (common/mod.rs) use the same interpreter.
PYTHON=""
for _p in \
        "$VIRTUAL_ENV/bin/python3" \
        "$VIRTUAL_ENV/bin/python" \
        "$(dirname "$SCRIPT_DIR")/venv/bin/python3" \
        "$(dirname "$SCRIPT_DIR")/venv/bin/python" \
        /agent/venv/bin/python3 \
        /agent/venv/bin/python \
        python3 python; do
    if command -v "$_p" >/dev/null 2>&1 && "$_p" -c "import sys; sys.exit(0)" 2>/dev/null; then
        PYTHON="$_p"; break
    fi
done
if [ -n "$PYTHON" ]; then
    export ZUNA_PYTHON="$PYTHON"
    ok "Python : $("$PYTHON" --version 2>&1)  ($PYTHON)"
else
    warn "No Python found (tried python3, python) — benchmark charts and weight download will be skipped"
fi

# ── Cargo / Rust ──────────────────────────────────────────────────────────────
. "$HOME/.cargo/env" 2>/dev/null || true
command -v cargo >/dev/null 2>&1 || die "cargo not found — install Rust: https://rustup.rs"

# Use /tmp for target so we avoid noexec or slow mounts
TARGET_DIR=/tmp/zuna-bench-target
EMBED_BIN=/tmp/zuna-bench-embed
INFER_BIN=/tmp/zuna-bench-infer

# ── Step 1: Build ──────────────────────────────────────────────────────────────
step "[1/4] Build"
build_example() {
    _name="$1"; _out="$2"
    if [ "$CACHED" = "1" ] && [ -f "$_out" ]; then
        ok "$_name: cached  ($_out)"
        return
    fi
    info "cargo build --release $FEATURES --example $_name"
    CARGO_TARGET_DIR="$TARGET_DIR" \
        cargo build --release $FEATURES --example "$_name" 2>&1 \
        | grep -E "^(error|warning\[|   Compiling|    Finished)" || true
    cp "$TARGET_DIR/release/examples/$_name" "$_out"
    chmod +x "$_out"
    ok "$_name  →  $_out"
}

build_example embed "$EMBED_BIN"
[ "$FULL" = "1" ] && build_example infer "$INFER_BIN"

# ── Step 2: Weights ────────────────────────────────────────────────────────────
step "[2/4] Weights"
WEIGHTS_FILE="${ZUNA_WEIGHTS:-}"
CONFIG_FILE="${ZUNA_CONFIG:-}"

if [ -z "$WEIGHTS_FILE" ] && [ -n "$PYTHON" ]; then
    info "Scanning HuggingFace cache …"
    # Prefix output lines with ZUNA_W: / ZUNA_C: so conda activation messages
    # (e.g. "Requirement already satisfied: numpy") can't pollute path parsing.
    _raw="$("$PYTHON" - 2>/dev/null <<'PYEOF'
import sys, os, pathlib

def emit(w, c):
    # Use prefixed markers — immune to conda stdout noise on the other lines
    sys.stdout.write(f"ZUNA_W:{w}\nZUNA_C:{c}\n")
    sys.stdout.flush()
    sys.exit(0)

# 1. Scan existing local cache
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for p in pathlib.Path(hf_home).rglob("model-00001-of-00001.safetensors"):
    c = p.parent / "config.json"
    if c.exists():
        emit(p, c)

# 2. Trigger download via huggingface_hub
try:
    from huggingface_hub import snapshot_download
    snap = pathlib.Path(snapshot_download("Zyphra/ZUNA"))
    emit(snap / "model-00001-of-00001.safetensors", snap / "config.json")
except Exception as e:
    sys.stderr.write(f"HF download: {e}\n")

sys.exit(1)
PYEOF
    )" || true
    WEIGHTS_FILE=$(printf '%s\n' "$_raw" | grep '^ZUNA_W:' | head -1 | sed 's/^ZUNA_W://')
    CONFIG_FILE=$(printf  '%s\n' "$_raw" | grep '^ZUNA_C:' | head -1 | sed 's/^ZUNA_C://')
fi

# Fallback: scan without Python
if [ -z "$WEIGHTS_FILE" ]; then
    WEIGHTS_FILE=$(find "$HOME/.cache/huggingface" -name "model-00001-of-00001.safetensors" 2>/dev/null | head -1)
    CONFIG_FILE=$(find  "$HOME/.cache/huggingface" -name "config.json" -path "*ZUNA*" 2>/dev/null | head -1)
fi

_py_hint="${PYTHON:-python3}"
[ -f "$WEIGHTS_FILE" ] || die "Weights not found.  Run: $_py_hint -c \"from huggingface_hub import snapshot_download; snapshot_download('Zyphra/ZUNA')\""
[ -f "$CONFIG_FILE"  ] || die "config.json not found alongside weights."

ok "weights : $WEIGHTS_FILE"
ok "config  : $CONFIG_FILE"

W_FLAG="--weights $WEIGHTS_FILE --config $CONFIG_FILE"

# ── Step 3: embed example ──────────────────────────────────────────────────────
step "[3/4] Encoder benchmark  (embed example)"

FIF="${ZUNA_FIF:-$SCRIPT_DIR/data/sample1_raw.fif}"
[ -f "$FIF" ] || die "FIF not found: $FIF  (set ZUNA_FIF=<path>)"
ok "FIF     : $FIF"

FIGURES="$SCRIPT_DIR/figures"
EMBEDDINGS_OUT=/tmp/zuna_bench_embeddings.safetensors
INPUTS_OUT=/tmp/zuna_bench_inputs.safetensors

info "Running embed example  (device=$DEVICE, $RUNS pass(es)) …"
info ""

# First pass: verbose summary + export pre-transformer inputs for Python comparison
"$EMBED_BIN" \
    $W_FLAG \
    --device "$DEVICE" \
    --fif "$FIF" \
    --output "$EMBEDDINGS_OUT" \
    --export-inputs "$INPUTS_OUT" \
    --figures "$FIGURES" \
    --verbose 2>&1

info ""

# Additional timing passes (silent) to get stable numbers
_pass=1
while [ $_pass -lt "$RUNS" ]; do
    _pass=$((_pass+1))
    info "Timing pass $_pass/$RUNS …"
    "$EMBED_BIN" \
        $W_FLAG \
        --device "$DEVICE" \
        --fif "$FIF" \
        --output "$EMBEDDINGS_OUT" \
        --no-charts \
        2>&1 | grep -E "^(TIMING|\[)" || true
done

ok "Embeddings  →  $EMBEDDINGS_OUT"
[ -f "$INPUTS_OUT" ] && ok "Bench inputs  →  $INPUTS_OUT"

# ── Step 4: Python benchmark ───────────────────────────────────────────────────
step "[4/4] bench_and_visualize.py"

if [ -z "$PYTHON" ]; then
    warn "Python not found — skipping bench_and_visualize.py"
else
    _no_py_flag=""
    [ "$NO_PYTHON" = "1" ] && _no_py_flag="--no-python-encoder"

    # Run bench_and_visualize.py; if the chosen interpreter fails for any
    # reason (e.g. missing numpy), retry once with the alternative one.
    _bv_ok=0
    if "$PYTHON" "$SCRIPT_DIR/scripts/bench_and_visualize.py" \
            $W_FLAG --fif "$FIF" --runs "$RUNS" --device "$DEVICE" \
            --embed-bin "$EMBED_BIN" $_no_py_flag 2>&1; then
        _bv_ok=1
    else
        _bv_exit=$?
        _alt_python=""
        case "$(basename "$PYTHON")" in
            python3*) command -v python  >/dev/null 2>&1 && _alt_python="$(command -v python)"  ;;
            python)   command -v python3 >/dev/null 2>&1 && _alt_python="$(command -v python3)" ;;
        esac
        if [ -n "$_alt_python" ]; then
            warn "'$(basename "$PYTHON") bench_and_visualize.py' failed (exit $_bv_exit) — retrying with $(basename "$_alt_python") …"
            "$_alt_python" "$SCRIPT_DIR/bench_and_visualize.py" \
                $W_FLAG --fif "$FIF" --runs "$RUNS" --device "$DEVICE" \
                --embed-bin "$EMBED_BIN" $_no_py_flag 2>&1
            _bv_ok=1
        else
            die "bench_and_visualize.py failed (exit $_bv_exit) and no alternative Python interpreter found"
        fi
    fi

    ok "Charts  →  $FIGURES/"
    ok "Data    →  $FIGURES/bench_data.json"
fi

# ── Optional: full encode+decode ───────────────────────────────────────────────
if [ "$FULL" = "1" ]; then
    step "[+] Full inference  (infer example — encode + diffuse + decode)"
    INFER_OUT=/tmp/zuna_bench_output.safetensors
    "$INFER_BIN" \
        $W_FLAG \
        --device "$DEVICE" \
        --fif "$FIF" \
        --output "$INFER_OUT" \
        --figures "$FIGURES" \
        --verbose 2>&1
    ok "Reconstruction  →  $INFER_OUT"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
step "Done"
ok "Backend : $BACKEND_LABEL"
ok "Figures : $FIGURES/"
ok "README  : $SCRIPT_DIR/README.md  (benchmark section auto-updated)"
info ""
info "View figures:"
info "  ls $FIGURES/*.png"
if [ "$FULL" = "1" ]; then
    info ""
    info "Inspect reconstruction:"
    info "  ${PYTHON:-python3} $SCRIPT_DIR/scripts/inspect_output.py $INFER_OUT"
fi
info ""
