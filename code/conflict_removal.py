"""
conflict_removal.py
Identifies and removes sequences that appear simultaneously
in IEDB positive and negative pools across different
experimental records (label conflicts).

Reads four folders of individual FASTA files, finds
sequences present in both positive and negative pools,
copies the clean remainder to a new output directory,
and verifies zero overlap remains.

Usage:
    python 1_conflict_removal.py

Paths are configured at the top of the script.
Expected output: 1,198 positives, 2,621 negatives.
"""

import os
import shutil


# ── paths ─────────────────────────────────────────────────────────────────────

CLEANED_DIR = "C:/jupyter/juan/AIP/data/data_fasta"
FINAL_DIR   = "C:/jupyter/juan/AIP/data/final_dataset"

FOLDERS = {
    "pos_train": os.path.join(CLEANED_DIR, "pos_train"),
    "neg_train": os.path.join(CLEANED_DIR, "neg_train"),
    "pos_test" : os.path.join(CLEANED_DIR, "pos_test"),
    "neg_test" : os.path.join(CLEANED_DIR, "neg_test"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_folder(folder):
    """Read all .fasta files in folder, return {filename: sequence}."""
    seq_map = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".fasta"):
            continue
        seq = ""
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    seq += line.upper()
        if seq:
            seq_map[fname] = seq
    return seq_map


def load_seqs(folder):
    """Return set of sequences from all .fasta files in folder."""
    seqs = set()
    for fname in os.listdir(folder):
        if not fname.endswith(".fasta"):
            continue
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(">"):
                    seqs.add(line.upper())
    return seqs


# ── load ──────────────────────────────────────────────────────────────────────

print("loading folders...")
pos_train = load_folder(FOLDERS["pos_train"])
neg_train = load_folder(FOLDERS["neg_train"])
pos_test  = load_folder(FOLDERS["pos_test"])
neg_test  = load_folder(FOLDERS["neg_test"])

print(f"pos_train: {len(pos_train)}")
print(f"neg_train: {len(neg_train)}")
print(f"pos_test : {len(pos_test)}")
print(f"neg_test : {len(neg_test)}")


# ── find conflicts ────────────────────────────────────────────────────────────
# a conflict is any sequence appearing in any positive folder
# AND any negative folder across any combination of records

all_pos = set(pos_train.values()) | set(pos_test.values())
all_neg = set(neg_train.values()) | set(neg_test.values())
conflicts = all_pos & all_neg

print(f"\nconflicting sequences found: {len(conflicts)}")


# ── remove conflicts ──────────────────────────────────────────────────────────

for split, seq_map in [
    ("pos_train", pos_train),
    ("neg_train", neg_train),
    ("pos_test",  pos_test),
    ("neg_test",  neg_test),
]:
    out_folder = os.path.join(FINAL_DIR, split)
    os.makedirs(out_folder, exist_ok=True)

    kept = removed = 0
    for fname, seq in seq_map.items():
        if seq in conflicts:
            removed += 1
            continue
        shutil.copy(
            os.path.join(FOLDERS[split], fname),
            os.path.join(out_folder, fname)
        )
        kept += 1

    print(f"{split:12}  kept={kept}  removed={removed}")


# ── verify zero overlap remains ───────────────────────────────────────────────

print("\nverifying final_dataset...")

fp_tr = load_seqs(os.path.join(FINAL_DIR, "pos_train"))
fn_tr = load_seqs(os.path.join(FINAL_DIR, "neg_train"))
fp_te = load_seqs(os.path.join(FINAL_DIR, "pos_test"))
fn_te = load_seqs(os.path.join(FINAL_DIR, "neg_test"))

print(f"pos_train : {len(fp_tr)}")
print(f"neg_train : {len(fn_tr)}")
print(f"pos_test  : {len(fp_te)}")
print(f"neg_test  : {len(fn_te)}")
print(f"total     : {len(fp_tr) + len(fn_tr) + len(fp_te) + len(fn_te)}")

print(f"\noverlap pos_train vs neg_train : {len(fp_tr & fn_tr)}")
print(f"overlap pos_test  vs neg_test  : {len(fp_te & fn_te)}")
print(f"overlap pos_train vs neg_test  : {len(fp_tr & fn_te)}")
print(f"overlap pos_test  vs neg_train : {len(fp_te & fn_tr)}")
print(f"overlap pos_train vs pos_test  : {len(fp_tr & fp_te)}")
print(f"overlap neg_train vs neg_test  : {len(fn_tr & fn_te)}")
