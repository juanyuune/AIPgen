# AIPgen: Denoised Training Enables Generative Discovery and Structure-Based Optimisation of Anti-Inflammatory Peptide Candidates

This repository contains the code and data for the AIPgen study. The core finding is that IEDB-derived AIP training datasets carry a previously unreported label noise problem — 432 sequences (26.5% of the positive pool) appear simultaneously as positive and negative across different experimental records, and IEDB-derived negatives are more similar to positives than positives are to each other in ESM-2 embedding space. We show this explains why all prior AIP classifiers plateau well below useful discrimination, and fix it through three-source negative reconstruction and systematic conflict removal before training.

The pipeline goes from raw IEDB sequences to a denoised benchmark, trains a multi-window CNN classifier, generates 618 novel AIP candidates with ProtGPT2, and carries the top candidate through BindCraft structure-based optimisation against TNF-α receptor.

---

## Repository structure

```
data/           9 FASTA files covering all train/test splits
supplementary/  432 conflicting sequence IDs (Supplementary Data 1)
weights/        trained MCNN model weights
scripts/        numbered pipeline scripts and evaluation code
models/         MCNN architecture definition
bindcraft/      BindCraft config used in the study
figures/        figure generation scripts
```

---

## Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

---

## Reproducing the main results

Run the scripts in order. Each step depends on the output of the previous one.

```bash
# Step 1 — conflict removal (reproduces Table 1)
python scripts/1_conflict_removal.py

# Step 2 — ESM-2 650M embedding
python scripts/2_embed_esm2.py

# Step 3 — ProtGPT2 conditional generation
python scripts/3_generate_protgpt2.py

# Step 4 — train MCNN classifier
python scripts/4_train_mcnn.py

# Step 5 — score synthetic sequences, select top 5 (reproduces Table 14)
python scripts/5_score_candidates.py

# Step 6 — ESMFold structural prediction
python scripts/6_esmfold_predict.py
```

For evaluation and figure reproduction:

```bash
# prior method comparison and ablation (Tables 5 and 6)
python scripts/evaluate.py

# cosine similarity analysis (Table 3)
python scripts/cosine_similarity_analysis.py

# Supplementary Figure S2 and conflict evidence
python figures/figureS2_conflict_analysis.py
```

BindCraft was run on Google Colab (T4 GPU) using the config in `bindcraft/bindcraft_config.json`. The target structure is PDB 1TNR chain A.

---

## Data sources

| Source | URL | Accessed |
|--------|-----|----------|
| IEDB | https://www.iedb.org | April 2026 |
| APD6 | https://aps.unmc.edu | April 2026 |
| SATPdb | http://crdd.osdd.net/raghava/satpdb/ | April 2026 |
| TNF-α receptor | PDB: 1TNR | — |

---

## Citation

To be added after publication.
