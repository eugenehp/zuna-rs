"""
ZUNA channel-padding benchmark — UMAP / t-SNE / PCA visualisation.

Handles all three benchmark modes (strategy / ablation / region / all).
Produces per-mode scatter plots (2-D, 3-D with legend) and a distribution
bar chart (also with legend).

Method fallback chain (stops at first that works):
  1. umap-learn   (pip install umap-learn)
  2. scikit-learn TSNE
  3. scikit-learn PCA
  4. numpy SVD PCA  ← always available, no extra deps

Usage (standalone):
  python3 scripts/umap_channel_bench.py \\
      data/channel_bench_strategy_embeddings.npy \\
      data/channel_bench_strategy_labels.npy \\
      figures/channel_bench_strategy_results.json \\
      figures/umap_strategy_2d.png \\
      figures/umap_strategy_3d.png \\
      30000
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.lines import Line2D

# ── Arguments ─────────────────────────────────────────────────────────────────
if len(sys.argv) != 7:
    sys.exit("Usage: umap_channel_bench.py "
             "emb.npy labels.npy results.json out_2d.png out_3d.png max_pts")
EMB_NPY, LBL_NPY, RESULTS_JSON, OUT_2D, OUT_3D = sys.argv[1:6]
MAX_PTS = int(sys.argv[6])

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"  Loading {EMB_NPY} …")
emb    = np.load(EMB_NPY).astype(np.float32)
labels = np.load(LBL_NPY).astype(np.int32)
with open(RESULTS_JSON) as fh:
    meta = json.load(fh)

config_names = meta["config_names"]
configs_data = meta.get("configurations", [])
mode         = meta.get("mode", "strategy").lower()
n_configs    = len(config_names)
print(f"  {emb.shape[0]:,} vectors × {emb.shape[1]} dims  |  {n_configs} configs  |  mode={mode}")

# ── Subsample if needed ────────────────────────────────────────────────────────
if len(emb) > MAX_PTS:
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(emb), MAX_PTS, replace=False))
    emb, labels = emb[idx], labels[idx]
    print(f"  Subsampled to {len(emb):,} vectors")

# ── Dimensionality reduction ───────────────────────────────────────────────────
def numpy_pca(X, n):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:n].T).astype(np.float32)

def reduce(n_components):
    try:
        import umap
        print(f"  UMAP → {n_components}D …")
        return umap.UMAP(n_components=n_components, n_neighbors=15,
                         min_dist=0.10, random_state=42, low_memory=True
                         ).fit_transform(emb), "UMAP"
    except ImportError: pass
    try:
        from sklearn.manifold import TSNE
        print(f"  t-SNE → {n_components}D …")
        return TSNE(n_components=n_components, perplexity=40,
                    random_state=42, n_jobs=-1).fit_transform(emb), "t-SNE"
    except ImportError: pass
    try:
        from sklearn.decomposition import PCA
        print(f"  sklearn PCA → {n_components}D …")
        return PCA(n_components=n_components, random_state=42).fit_transform(emb), "PCA (sklearn)"
    except ImportError: pass
    print(f"  numpy PCA → {n_components}D …")
    return numpy_pca(emb, n_components), "PCA (numpy)"

coords2, method2 = reduce(2)
coords3, method3 = reduce(3)

# ── Colour / marker palettes ───────────────────────────────────────────────────
CH_GROUPS  = ["12ch", "10ch", "8ch", "6ch", "4ch", "2ch"]
CH_COLOURS = dict(zip(CH_GROUPS, plt.cm.tab10(np.linspace(0, 0.6, len(CH_GROUPS)))))

STRATEGY_STYLE = {          # label → (marker, size, fixed_colour_or_None)
    "baseline":        ("*",  12, "#222222"),
    "zero_pad":        ("o",   8, "#999999"),
    "clone_nearest":   ("s",   8, None),
    "clone_fp1":       ("^",   8, None),
    "noisy_clone":     ("D",   7, None),
    "xyz_jitter":      ("P",   9, None),
    "interp_weighted": ("h",   9, None),
    "mirror":          ("<",   8, None),
    "mean_ref":        (">",   8, None),
    "native":          ("X",  10, None),   # bold X — "no padding, real channels only"
}

ALL_12 = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8"]
ABLATION_COLOURS = dict(zip(
    [f"drop_{c}" for c in ALL_12],
    plt.cm.tab20(np.linspace(0, 1, len(ALL_12))),
))

REGION_KEYS = ["frontal_drop", "central_drop", "posterior_drop",
               "left_hemi_drop", "right_hemi_drop"]
REGION_LABELS = ["frontal", "central", "posterior", "left hemi", "right hemi"]
REGION_COLOURS = dict(zip(REGION_KEYS, plt.cm.Set1(np.linspace(0, 0.8, len(REGION_KEYS)))))

# ── Per-config style lookup ────────────────────────────────────────────────────
def config_style(name):
    """Return (colour, marker, markersize) for one config label."""
    if name.startswith("drop_"):
        return ABLATION_COLOURS.get(name, (0.5,0.5,0.5,1.0)), "o", 8
    for rk in REGION_KEYS:
        if name.startswith(rk):
            suffix = name[len(rk)+1:]
            mk, ms, _ = STRATEGY_STYLE.get(suffix, ("o", 8, None))
            return REGION_COLOURS[rk], mk, ms
    for grp in CH_GROUPS:
        if name.startswith(grp) or name == "12ch_baseline":
            colour = CH_COLOURS.get(grp if name != "12ch_baseline" else "12ch")
            suffix = name[len(grp)+1:] if name != "12ch_baseline" else "baseline"
            mk, ms, fixed = STRATEGY_STYLE.get(suffix, ("o", 8, None))
            return (fixed if fixed else colour), mk, ms
    return (0.5,0.5,0.5,1.0), "o", 7

# Pre-compute styles once
styles = {name: config_style(name) for name in config_names}

# ── Legend builders ────────────────────────────────────────────────────────────
def build_legend_handles():
    """Return list of legend handles appropriate for the current mode."""
    handles = []
    if mode in ("strategy", "all"):
        handles += [Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=CH_COLOURS[g], markersize=10,
                           label=f"{g} channels")
                    for g in CH_GROUPS]
        handles.append(Line2D([0],[0], color="none", label=""))   # spacer
        handles += [Line2D([0],[0], marker=mk, color="#555555",
                           markersize=ms, label=s.replace("_", " "))
                    for s, (mk, ms, _) in STRATEGY_STYLE.items()]

    elif mode == "ablation":
        handles += [Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=ABLATION_COLOURS[f"drop_{c}"],
                           markersize=10, label=f"drop {c}")
                    for c in ALL_12]

    elif mode == "region":
        handles += [Line2D([0],[0], marker="o", color="w",
                           markerfacecolor=REGION_COLOURS[rk],
                           markersize=10, label=lbl)
                    for rk, lbl in zip(REGION_KEYS, REGION_LABELS)]
        handles.append(Line2D([0],[0], color="none", label=""))
        handles += [Line2D([0],[0], marker=mk, color="#555555",
                           markersize=ms,
                           label=s.replace("_", " "))
                    for s, (mk, ms, _) in STRATEGY_STYLE.items()
                    if s in ("zero_pad","clone_nearest","mirror","mean_ref")]
    return handles

legend_handles = build_legend_handles()

# ── Scatter helpers ────────────────────────────────────────────────────────────
def scatter_2d(ax, x, y):
    for ci, cname in enumerate(config_names):
        mask = labels == ci
        if not mask.any(): continue
        colour, mk, ms = styles[cname]
        ax.scatter(x[mask], y[mask], c=[colour], marker=mk,
                   s=ms**1.5, alpha=0.45, linewidths=0)

def scatter_3d(ax, x, y, z):
    for ci, cname in enumerate(config_names):
        mask = labels == ci
        if not mask.any(): continue
        colour, mk, ms = styles[cname]
        ax.scatter(x[mask], y[mask], z[mask], c=[colour], marker=mk,
                   s=ms**1.4, alpha=0.40, linewidths=0, depthshade=True)

# ── 2-D figure ─────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 10))
scatter_2d(ax2, coords2[:, 0], coords2[:, 1])
fig2.suptitle(
    f"ZUNA EEG — Channel Benchmark  [{mode} mode, {method2}]\n"
    f"{len(emb):,} vectors  ·  {n_configs} configurations",
    fontsize=13,
)
ax2.set_xlabel(f"{method2} dim 1", fontsize=10)
ax2.set_ylabel(f"{method2} dim 2", fontsize=10)

# Legend: place outside the axes on the right so it never overlaps the data
leg2 = ax2.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    borderaxespad=0,
    fontsize=9,
    framealpha=0.95,
    ncol=1,
    title={"strategy": "colour = ch-count  ·  marker = strategy",
           "ablation":  "colour = dropped channel",
           "region":    "colour = region  ·  marker = strategy",
           }.get(mode, ""),
    title_fontsize=8,
)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig(OUT_2D, dpi=150, bbox_inches="tight")
print(f"  Saved 2-D → {OUT_2D}")
plt.close(fig2)

# ── 3-D figure ─────────────────────────────────────────────────────────────────
# Layout: 2×2 subplots + a dedicated legend axes on the right
views = [
    ("Perspective",      30, -60),
    ("Top (axial)",      90, -90),
    ("Front (coronal)",   0, -90),
    ("Side (sagittal)",   0,   0),
]

fig3 = plt.figure(figsize=(22, 16))
fig3.suptitle(
    f"ZUNA EEG — Channel Benchmark  [{mode} mode, {method3}, 3-D]\n"
    f"{len(emb):,} vectors  ·  {n_configs} configurations",
    fontsize=13, y=0.99,
)

# Use GridSpec: 2 rows × 3 cols — left 2 cols = 4 scatter panels, right col = legend
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig3, width_ratios=[1, 1, 0.28],
              hspace=0.30, wspace=0.15)

panel_positions = [(0,0), (0,1), (1,0), (1,1)]
for pi, ((row, col), (title, elev, azim)) in enumerate(zip(panel_positions, views)):
    ax3 = fig3.add_subplot(gs[row, col], projection="3d")
    ax3.set_title(title, fontsize=11, pad=4)
    scatter_3d(ax3, coords3[:,0], coords3[:,1], coords3[:,2])
    ax3.set_xlabel("dim 1", fontsize=7, labelpad=2)
    ax3.set_ylabel("dim 2", fontsize=7, labelpad=2)
    ax3.set_zlabel("dim 3", fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)
    ax3.view_init(elev=elev, azim=azim)

# Dedicated legend axes (right column, spanning both rows)
leg_ax = fig3.add_subplot(gs[:, 2])
leg_ax.axis("off")
leg_ax.legend(
    handles=legend_handles,
    loc="center left",
    fontsize=10,
    framealpha=0.95,
    ncol=1,
    title={"strategy": "colour = ch-count\nmarker = strategy",
           "ablation":  "colour = dropped channel",
           "region":    "colour = region\nmarker = strategy",
           }.get(mode, ""),
    title_fontsize=9,
    borderaxespad=0,
)

plt.savefig(OUT_3D, dpi=150, bbox_inches="tight")
print(f"  Saved 3-D → {OUT_3D}")
plt.close(fig3)

# ── Distribution bar chart ─────────────────────────────────────────────────────
if configs_data:
    bar_width = max(14, n_configs * 0.55)
    # Add extra right margin for the legend
    fig4, ax4 = plt.subplots(figsize=(bar_width + 4, 7))

    x_pos  = np.arange(n_configs)
    means  = np.array([c["stats"]["mean"] for c in configs_data])
    stds   = np.array([c["stats"]["std"]  for c in configs_data])
    colors = [styles[c["name"]][0] for c in configs_data]

    ax4.bar(x_pos, stds, color=colors, alpha=0.70, zorder=2)
    ax4.errorbar(x_pos, means, yerr=stds, fmt="none",
                 ecolor="#222222", elinewidth=0.8, capsize=3, zorder=3)
    ax4.axhline(0, color="k", linewidth=0.6, linestyle="--", zorder=1)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([c["name"] for c in configs_data],
                        rotation=55, fontsize=7, ha="right")
    ax4.set_ylabel("Embedding value  (bar height = std, whisker = mean ± std)", fontsize=10)
    ax4.set_title(
        f"ZUNA Embeddings — mean ± std per config  [{mode} mode]",
        fontsize=12,
    )
    ax4.grid(axis="y", alpha=0.3, zorder=0)

    # Build distribution-specific legend — use colour patches (no markers needed)
    if mode in ("strategy", "all"):
        dist_handles = [mpatches.Patch(facecolor=CH_COLOURS[g], label=f"{g} channels")
                        for g in CH_GROUPS]
        leg_title = "colour = channel count"
    elif mode == "ablation":
        dist_handles = [mpatches.Patch(facecolor=ABLATION_COLOURS[f"drop_{c}"],
                                       label=f"drop {c}")
                        for c in ALL_12]
        leg_title = "colour = dropped channel"
    else:  # region
        dist_handles = [mpatches.Patch(facecolor=REGION_COLOURS[rk], label=lbl)
                        for rk, lbl in zip(REGION_KEYS, REGION_LABELS)]
        leg_title = "colour = region"

    ax4.legend(
        handles=dist_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        fontsize=10,
        framealpha=0.95,
        title=leg_title,
        title_fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    dist_path = OUT_2D.replace("_2d.png", "_distributions.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"  Saved distributions → {dist_path}")
    plt.close(fig4)

print("  Visualisation complete.")
