# AIPgen: Denoised Training Enables Generative Discovery and
# Structure-Based Optimisation of Anti-Inflammatory Peptide Candidates

## Overview
AIPgen is a generative anti-inflammatory peptide (AIP) discovery
pipeline that resolves label noise in IEDB-derived training data
through three-source negative reconstruction and systematic conflict
removal, trains a Multiple Window CNN classifier on the denoised
benchmark, generates 618 novel AIP candidates using ProtGPT2, and
performs structure-based optimisation of the top candidate against
TNF-α receptor using BindCraft.

## Repository Structure
data/          — all 9 FASTA dataset files
supplementary/ — 432 conflicting sequence IDs (Supplementary Data 1)
weights/       — trained MCNN model weights
scripts/       — numbered pipeline scripts + evaluation utilities
models/        — MCNN architecture
bindcraft/     — BindCraft configuration used in the study
figures/       — scripts to reproduce all main figures

## Installation
pip install -r requirements.txt
or
conda env create -f environment.yml

## Reproducing Main Results

Step 1 — Conflict removal (reproduces Table 1)
  python scripts/1_conflict_removal.py

Step 2 — Embed sequences with ESM-2
  python scripts/2_embed_esm2.py

Step 3 — Generate synthetic candidates
  python scripts/3_generate_protgpt2.py

Step 4 — Train MCNN classifier
  python scripts/4_train_mcnn.py

Step 5 — Score and select top candidates (reproduces Table 14)
  python scripts/5_score_candidates.py

Step 6 — Structural prediction
  python scripts/6_esmfold_predict.py

Evaluate — Reproduces Table 5 and Table 6
  python scripts/evaluate.py

Cosine similarity — Reproduces Table 3
  python scripts/cosine_similarity_analysis.py

## Citation
[to be added after publication]

## Data Sources
IEDB:   https://www.iedb.org          (accessed April 2026)
APD6:   https://aps.unmc.edu          (accessed April 2026)
SATPdb: http://crdd.osdd.net/raghava/satpdb/ (accessed April 2026)
Target: PDB 1TNR (TNF-α receptor, chain A)
