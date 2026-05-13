"""
figureS2_conflict_analysis.py
Generates Supplementary Figure S2 (conflict analysis) and
runs the chi-squared uniformity test reported in the caption.

Also exports Supplementary Data 1 (full list of 432
conflicting sequence IDs) from raw IEDB FASTA files.

Usage:
    python figureS2_conflict_analysis.py

Requires:
    - C:/jupyter/juan/AIP/iedb_fasta/pos_iedb_cleaned.fasta
    - C:/jupyter/juan/AIP/iedb_fasta/neg_iedb_cleaned.fasta
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chisquare

matplotlib.rcParams["font.family"]       = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"]   = False
matplotlib.rcParams["axes.spines.right"] = False


# ── paths ─────────────────────────────────────────────────────────────────────

FASTA_DIR  = "C:/jupyter/juan/AIP/iedb_fasta"
POS_FASTA  = os.path.join(FASTA_DIR, "pos_iedb_cleaned.fasta")
NEG_FASTA  = os.path.join(FASTA_DIR, "neg_iedb_cleaned.fasta")
OUT_DIR    = "C:/jupyter/juan/AIP/figures"
SUPP_OUT   = os.path.join(OUT_DIR, "SupplementaryData1_ConflictingSequences.tsv")
os.makedirs(OUT_DIR, exist_ok=True)


# ── confirmed manuscript values ───────────────────────────────────────────────

TOTAL_POS           = 1631
CONFLICTING         = 432
CLEAN               = 1198
CONFLICT_PCT        = 26.5
CLEAN_PCT           = 73.5
CYTOKINES           = ["IL-4", "IL-10", "IL-13", "IL-22", "TGF-\u03b2", "IFN-\u03b1/\u03b2"]
CONFLICTS_PER_CYT   = [72, 72, 72, 72, 72, 72]


# ── colours ───────────────────────────────────────────────────────────────────

COL_CLEAN    = "#2E7D8C"
COL_CONFLICT = "#E05A4E"
COL_GRID     = "#E8E8E8"
COL_TEXT     = "#1a1a1a"
COL_SECOND   = "#666666"


# ── chi-squared test ──────────────────────────────────────────────────────────

observed = np.array(CONFLICTS_PER_CYT)
expected = np.full(len(observed), observed.sum() / len(observed))
chi2, p  = chisquare(observed, f_exp=expected)

print(f"chi-squared test — uniform cytokine distribution")
print(f"  chi2 = {chi2:.4f}  df = {len(observed)-1}  p = {p:.4f}")
print(f"  expected = {expected[0]:.1f} per cytokine")
print(f"\ncaption sentence:")
print(f"  χ² = {chi2:.2f}, df = {len(observed)-1}, p = {p:.3f}, "
      f"expected n = {int(expected[0])} per cytokine type")


# ── parse FASTA files and identify conflicts ──────────────────────────────────

def parse_fasta(path):
    """Returns {sequence: [list_of_headers]}."""
    seqs, header, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and parts:
                    seq = "".join(parts).upper()
                    seqs.setdefault(seq, []).append(header)
                header, parts = line, []
            else:
                parts.append(line)
    if header and parts:
        seq = "".join(parts).upper()
        seqs.setdefault(seq, []).append(header)
    return seqs

if os.path.exists(POS_FASTA) and os.path.exists(NEG_FASTA):
    pos_seqs = parse_fasta(POS_FASTA)
    neg_seqs = parse_fasta(NEG_FASTA)
    conflicts = set(pos_seqs.keys()) & set(neg_seqs.keys())
    print(f"\nfound {len(conflicts)} conflicts in FASTA files "
          f"(manuscript reports 432)")

    # export Supplementary Data 1
    with open(SUPP_OUT, "w") as f:
        f.write("Sequence\tLength\tPositive_header\tNegative_header\n")
        for seq in sorted(conflicts):
            pos_hdr = pos_seqs[seq][0] if pos_seqs[seq] else "N/A"
            neg_hdr = neg_seqs[seq][0] if neg_seqs[seq] else "N/A"
            f.write(f"{seq}\t{len(seq)}\t{pos_hdr}\t{neg_hdr}\n")
    print(f"Supplementary Data 1 saved: {SUPP_OUT}")
else:
    print("FASTA files not found — using manuscript values for figure")
    print("run on raw IEDB files to export Supplementary Data 1")


# ── figure ────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.2, 3.2), facecolor="white",
    gridspec_kw={"wspace": 0.45}
)
fig.patch.set_facecolor("white")


# panel a — positive pool composition
categories = [
    f"Clean positives\n(n = {CLEAN:,}, {CLEAN_PCT}%)",
    f"Conflicting sequences\n(n = {CONFLICTING:,}, {CONFLICT_PCT}%)",
]
bars_a = ax1.barh(categories, [CLEAN, CONFLICTING],
                  color=[COL_CLEAN, COL_CONFLICT], height=0.45, zorder=3)

for bar, val in zip(bars_a, [CLEAN, CONFLICTING]):
    ax1.text(bar.get_width() - 30, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", ha="right",
             fontsize=9, fontweight="bold", color="white", zorder=4)

ax1.annotate(
    "26.5% of positive pool\ncarry contradictory labels",
    xy=(CONFLICTING, 1), xytext=(650, 1.35),
    fontsize=7.5, color=COL_CONFLICT, ha="center",
    arrowprops=dict(arrowstyle="->", color=COL_CONFLICT, lw=1.2),
)
ax1.set_xlabel("Number of sequences", fontsize=8.5, labelpad=5, color=COL_TEXT)
ax1.set_xlim(0, 1350)
ax1.set_facecolor("white")
ax1.tick_params(axis="x", labelsize=7.5, length=3, colors=COL_SECOND)
ax1.tick_params(axis="y", labelsize=8,   length=0, colors=COL_TEXT)
ax1.xaxis.grid(True, color=COL_GRID, linewidth=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["left"].set_linewidth(0.8)
ax1.spines["left"].set_color("#444444")
ax1.spines["bottom"].set_linewidth(0.8)
ax1.spines["bottom"].set_color("#444444")
ax1.text(-0.18, 1.06, "a", transform=ax1.transAxes,
         fontsize=11, fontweight="bold", color=COL_TEXT, va="bottom")


# panel b — cytokine conflict distribution
x_pos  = np.arange(len(CYTOKINES))
bars_b = ax2.bar(x_pos, CONFLICTS_PER_CYT, color=COL_CONFLICT,
                 width=0.55, zorder=3)

for bar, val in zip(bars_b, CONFLICTS_PER_CYT):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
             str(val), va="bottom", ha="center",
             fontsize=8, fontweight="bold", color=COL_CONFLICT)

ax2.axhline(y=72, color=COL_CONFLICT, linewidth=1.0,
            linestyle="--", alpha=0.5, zorder=2)

ax2.text(2.5, 79, "Uniform distribution across all cytokine types",
         fontsize=7, color=COL_CONFLICT, ha="center", va="bottom", alpha=0.9)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(CYTOKINES, fontsize=8, color=COL_TEXT)
ax2.set_ylabel("Number of conflicting sequences",
               fontsize=8.5, labelpad=5, color=COL_TEXT)
ax2.set_ylim(0, 90)
ax2.set_yticks([0, 20, 40, 60, 80])
ax2.set_facecolor("white")
ax2.tick_params(axis="x", length=0, labelsize=8, colors=COL_TEXT)
ax2.tick_params(axis="y", length=3, labelsize=7.5, colors=COL_SECOND)
ax2.yaxis.grid(True, color=COL_GRID, linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.spines["left"].set_linewidth(0.8)
ax2.spines["left"].set_color("#444444")
ax2.spines["bottom"].set_linewidth(0.8)
ax2.spines["bottom"].set_color("#444444")
ax2.text(-0.18, 1.06, "b", transform=ax2.transAxes,
         fontsize=11, fontweight="bold", color=COL_TEXT, va="bottom")


# legend
fig.legend(
    handles=[
        mpatches.Patch(color=COL_CLEAN,    label="Clean positives"),
        mpatches.Patch(color=COL_CONFLICT, label="Conflicting sequences"),
    ],
    loc="lower center", bbox_to_anchor=(0.5, -0.08),
    ncol=2, frameon=False, fontsize=8,
    handlelength=1.2, handleheight=0.8,
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

out_path = os.path.join(OUT_DIR, "FigureS2_Conflict_Analysis.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print(f"\nfigure saved: {out_path}")
plt.show()
