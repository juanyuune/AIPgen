"""
cosine_similarity_analysis.py
Reproduces Table 3 and Figure 1 from the manuscript.

Computes mean pairwise cosine similarity between ESM-2
embeddings of positive sequences and each of the three
negative sources (IEDB, SATPdb, APD6), plus within-class
similarities. Generates t-SNE projection and bar chart.

Usage:
    python cosine_similarity_analysis.py

Requires:
    - ESM-2 .esm2 embedding files in emb/esm2/pos_train/
      and emb/esm2/neg_train/
    - Source FASTA files for label assignment
    - pip install scikit-learn biopython matplotlib scipy
"""

import os
import numpy as np
import numpy.random as rnd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
from Bio import SeqIO
import sklearn


# ── paths ─────────────────────────────────────────────────────────────────────

BASE      = "C:/jupyter/juan/AIP"
EMB_POS   = os.path.join(BASE, "emb/esm2/pos_train")
EMB_NEG   = os.path.join(BASE, "emb/esm2/neg_train")
FASTA_POS = os.path.join(BASE, "data/data_fasta/pos_train")
FASTA_NEG = os.path.join(BASE, "data/data_fasta/neg_train")
OUT_DIR   = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FASTA_SOURCES = {
    "Positive (AIP)":  [os.path.join(BASE, "iedb_fasta/pos_iedb_cleaned.fasta")],
    "IEDB negative":   [os.path.join(BASE, "iedb_fasta/neg_iedb_cleaned.fasta")],
    "SATPdb negative": [
        os.path.join(BASE, "iedb_fasta/antifungal_cleaned.fasta"),
        os.path.join(BASE, "iedb_fasta/antihypertensive_cleaned.fasta"),
        os.path.join(BASE, "iedb_fasta/antiparasitic_cleaned.fasta"),
    ],
    "APD6 negative":   [os.path.join(BASE, "iedb_fasta/naturalAMPs_cleaned.fasta")],
}

APD6_FASTA   = os.path.join(BASE, "iedb_fasta/naturalAMPs_cleaned.fasta")
SATPDB_FILES = [
    os.path.join(BASE, "iedb_fasta/antifungal_cleaned.fasta"),
    os.path.join(BASE, "iedb_fasta/antihypertensive_cleaned.fasta"),
    os.path.join(BASE, "iedb_fasta/antiparasitic_cleaned.fasta"),
]


# ── figure style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         7,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth":    0.5,
    "lines.linewidth":   0.8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


# ── build sequence → source label lookup ──────────────────────────────────────

seq_to_label = {}
for label, files in FASTA_SOURCES.items():
    for fpath in files:
        if os.path.exists(fpath):
            for rec in SeqIO.parse(fpath, "fasta"):
                seq_to_label[str(rec.seq).upper().strip()] = label
        else:
            print(f"warning: {fpath} not found")

print("source sequences loaded:")
for label in FASTA_SOURCES:
    n = sum(1 for v in seq_to_label.values() if v == label)
    print(f"  {label}: {n}")


# ── load FASTA sequences ──────────────────────────────────────────────────────

def load_fasta_folder(folder):
    seqs = {}
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if os.path.isdir(fpath):
            continue
        try:
            for rec in SeqIO.parse(fpath, "fasta"):
                seqs[str(rec.id).strip()] = str(rec.seq).upper().strip()
        except Exception:
            seq_id = os.path.splitext(fname)[0]
            try:
                with open(fpath) as f:
                    seq = "".join(l for l in f if not l.startswith(">")).strip()
                if seq:
                    seqs[seq_id] = seq.upper()
            except Exception:
                continue
    return seqs

pos_fasta = load_fasta_folder(FASTA_POS)
neg_fasta = load_fasta_folder(FASTA_NEG)
print(f"\npos_train: {len(pos_fasta)}  neg_train: {len(neg_fasta)}")


# ── load embeddings ───────────────────────────────────────────────────────────

def load_emb(path):
    try:
        arr = np.load(path, allow_pickle=True).astype(np.float32).squeeze()
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    except Exception:
        return None


def load_emb_folder(folder, fasta_seqs):
    embs, labels, ids = [], [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".esm2"):
            continue
        seq_id = os.path.splitext(fname)[0]
        vec    = load_emb(os.path.join(folder, fname))
        if vec is None or vec.ndim != 1:
            continue
        seq   = fasta_seqs.get(seq_id, "")
        label = seq_to_label.get(seq, "Unknown")
        embs.append(vec)
        labels.append(label)
        ids.append(seq_id)
    return np.array(embs), labels, ids

print("\nloading embeddings...")
pos_emb, pos_labels, _ = load_emb_folder(EMB_POS, pos_fasta)
neg_emb, neg_labels, _ = load_emb_folder(EMB_NEG, neg_fasta)
print(f"  pos: {len(pos_emb)}  neg: {len(neg_emb)}")

label_counts = {}
for l in neg_labels:
    label_counts[l] = label_counts.get(l, 0) + 1
print("negative source labels:")
for l, c in sorted(label_counts.items()):
    print(f"  {l}: {c}")


# ── sample and compute similarities ──────────────────────────────────────────

rnd.seed(42)
N = 200

def sample_label(emb, labels, target, n):
    idx = [i for i, l in enumerate(labels) if l == target]
    if not idx:
        return np.zeros((0, emb.shape[1]))
    idx = rnd.choice(idx, min(n, len(idx)), replace=False)
    return emb[idx]

pos_s    = pos_emb[rnd.choice(len(pos_emb), min(N, len(pos_emb)), replace=False)]
iedb_s   = sample_label(neg_emb, neg_labels, "IEDB negative",   N)
satpdb_s = sample_label(neg_emb, neg_labels, "SATPdb negative", N)
apd6_s   = sample_label(neg_emb, neg_labels, "APD6 negative",   N)
neg_s    = neg_emb[rnd.choice(len(neg_emb), min(N, len(neg_emb)), replace=False)]

def mean_cos(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.mean(cosine_similarity(a, b)))

pos_pos    = mean_cos(pos_s, pos_s)
neg_neg    = mean_cos(neg_s, neg_s)
pos_iedb   = mean_cos(pos_s, iedb_s)
pos_satpdb = mean_cos(pos_s, satpdb_s)
pos_apd6   = mean_cos(pos_s, apd6_s)

print(f"\n{'='*55}")
print("TABLE 3 — cosine similarity values")
print(f"{'='*55}")
print(f"  Positive vs positive (within-class) : {pos_pos:.4f}")
print(f"  Negative vs negative (within-class) : {neg_neg:.4f}")
print(f"  Positive vs IEDB negative           : {pos_iedb:.4f}")
print(f"  Positive vs SATPdb negative         : {pos_satpdb:.4f}")
print(f"  Positive vs APD6 negative           : {pos_apd6:.4f}")


# ── t-SNE ─────────────────────────────────────────────────────────────────────

groups = []
for label, arr in [
    ("Positive (AIP)",  pos_s),
    ("IEDB negative",   iedb_s),
    ("SATPdb negative", satpdb_s),
    ("APD6 negative",   apd6_s),
]:
    if len(arr) > 0:
        groups.append((label, arr))

X          = np.vstack([g[1] for g in groups])
labels_all = [l for g in groups for l in [g[0]] * len(g[1])]

print(f"\nrunning t-SNE on {len(X)} sequences...")
sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
if sk_ver >= (1, 2):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=2000, learning_rate="auto", init="pca")
else:
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X)
print("t-SNE done")


# ── plot helpers ──────────────────────────────────────────────────────────────

STYLE = {
    "Positive (AIP)":  {"color": "#2166AC", "marker": "o", "size": 18, "alpha": 0.70, "zorder": 4},
    "IEDB negative":   {"color": "#D6604D", "marker": "^", "size": 18, "alpha": 0.65, "zorder": 3},
    "SATPdb negative": {"color": "#4DAC26", "marker": "s", "size": 16, "alpha": 0.60, "zorder": 2},
    "APD6 negative":   {"color": "#E08214", "marker": "D", "size": 16, "alpha": 0.60, "zorder": 1},
}
PLOT_ORDER = ["APD6 negative", "SATPdb negative", "IEDB negative", "Positive (AIP)"]

legend_handles = [
    Line2D([0], [0], marker=STYLE[l]["marker"], color="w",
           markerfacecolor=STYLE[l]["color"], markersize=4, label=l, alpha=0.9)
    for l in ["Positive (AIP)", "IEDB negative", "SATPdb negative", "APD6 negative"]
]


def draw_hull(ax, points, color):
    if len(points) < 4:
        return
    try:
        hull     = ConvexHull(points)
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull_pts, 0], points[hull_pts, 1], color=color, alpha=0.10, zorder=0)
        ax.plot(points[hull_pts, 0], points[hull_pts, 1], color=color, alpha=0.25,
                linewidth=0.6, zorder=0, linestyle="--")
    except Exception:
        pass


def draw_tsne(ax):
    for label in PLOT_ORDER:
        idx = [i for i, l in enumerate(labels_all) if l == label]
        if not idx:
            continue
        s = STYLE[label]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c=s["color"], marker=s["marker"],
                   s=s["size"], alpha=s["alpha"], linewidths=0.2, edgecolors="white",
                   zorder=s["zorder"], rasterized=True)
    for label, color in [("Positive (AIP)", "#2166AC"),
                         ("IEDB negative",  "#D6604D"),
                         ("APD6 negative",  "#E08214")]:
        idx = [i for i, l in enumerate(labels_all) if l == label]
        if len(idx) >= 4:
            draw_hull(ax, X_2d[idx], color)
    ax.set_xlabel("t-SNE 1", fontsize=7, labelpad=3)
    ax.set_ylabel("t-SNE 2", fontsize=7, labelpad=3)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=5.5,
              framealpha=0.9, edgecolor="0.8", handletextpad=0.4,
              borderpad=0.5, labelspacing=0.3)


comp_labels = ["Pos-Pos\n(within)", "Neg-Neg\n(within)",
               "Pos-IEDB\nneg", "Pos-SATPdb\nneg", "Pos-APD6\nneg"]
values     = [pos_pos, neg_neg, pos_iedb, pos_satpdb, pos_apd6]
bar_colors = ["#2166AC", "#AAAAAA", "#D6604D", "#4DAC26", "#E08214"]


def draw_bars(ax):
    bars = ax.bar(range(len(values)), values, color=bar_colors,
                  edgecolor="white", linewidth=0.5, width=0.62, zorder=3)
    bars[2].set_hatch("///")
    bars[2].set_edgecolor("#B03020")
    bars[2].set_linewidth(0.8)
    ax.axhline(y=pos_pos, color="#2166AC", linestyle="--", linewidth=0.8,
               alpha=0.8, zorder=2, label=f"Within-class positive ({pos_pos:.4f})")
    if pos_iedb > pos_pos:
        diff = pos_iedb - pos_pos
        ax.annotate(f"+{diff:.4f}", xy=(2, pos_iedb + 0.0003), xytext=(2, pos_iedb + 0.008),
                    ha="center", fontsize=5.5, color="#B03020", fontweight="bold")
        ax.annotate("", xy=(2, pos_iedb + 0.0005), xytext=(2, pos_iedb + 0.007),
                    arrowprops=dict(arrowstyle="->", color="#B03020", lw=0.8))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0006, f"{val:.4f}",
                ha="center", va="bottom", fontsize=5, color="0.2")
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, fontsize=5.5)
    ax.set_ylabel("Mean cosine similarity", fontsize=7, labelpad=4)
    ax.set_ylim(min(values) - 0.018, max(values) + 0.025)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5, framealpha=0.9, edgecolor="0.8",
              loc="lower right", handlelength=1.5)


# ── save figures ──────────────────────────────────────────────────────────────

# panel a — t-SNE
fig_a, ax_a = plt.subplots(figsize=(3.46, 3.20))
draw_tsne(ax_a)
ax_a.text(-0.08, 1.05, "a", transform=ax_a.transAxes,
          fontsize=9, fontweight="bold", va="top")
plt.tight_layout(pad=0.5)
path_a = os.path.join(OUT_DIR, "Figure1a_tSNE.png")
fig_a.savefig(path_a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig_a)
print(f"saved: {path_a}")

# panel b — cosine similarity bar chart
fig_b, ax_b = plt.subplots(figsize=(3.46, 3.20))
draw_bars(ax_b)
ax_b.text(-0.12, 1.05, "b", transform=ax_b.transAxes,
          fontsize=9, fontweight="bold", va="top")
plt.tight_layout(pad=0.5)
path_b = os.path.join(OUT_DIR, "Figure1b_CosineSimilarity.png")
fig_b.savefig(path_b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig_b)
print(f"saved: {path_b}")

# combined — Nature double column (183mm = 7.09in)
fig_c = plt.figure(figsize=(7.09, 3.15))
gs    = gridspec.GridSpec(1, 2, figure=fig_c, wspace=0.40,
                          left=0.07, right=0.97, top=0.92, bottom=0.12)
ax1 = fig_c.add_subplot(gs[0, 0])
ax2 = fig_c.add_subplot(gs[0, 1])
draw_tsne(ax1)
ax1.text(-0.08, 1.07, "a", transform=ax1.transAxes,
         fontsize=9, fontweight="bold", va="top")
draw_bars(ax2)
ax2.text(-0.14, 1.07, "b", transform=ax2.transAxes,
         fontsize=9, fontweight="bold", va="top")
path_c = os.path.join(OUT_DIR, "Figure1_Combined_ESM2_Analysis.png")
fig_c.savefig(path_c, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig_c)
print(f"saved: {path_c}")
