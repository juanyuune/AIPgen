"""
generate_protgpt2.py
ProtGPT2 conditional generation for AIP candidates.

Each real positive training sequence is used as a prompt.
One novel continuation is generated per prompt, then filtered
through 5 quality gates + 40% CD-HIT deduplication before
adding to the synthetic training set.

Usage:
    python 3_generate_protgpt2.py \
        -in  C:/jupyter/juan/AIP/data/data_fasta/pos_train \
        -out C:/jupyter/juan/AIP/iedb_fasta/synthetic_train \
        -n   1 -min 5 -max 50 -task AIP

Requirements:
    pip install transformers torch tqdm
"""

import os
import csv
import math
import logging
import argparse
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("-in",   "--input_folder",   required=True)
parser.add_argument("-out",  "--output_folder",  required=True)
parser.add_argument("-n",    "--n_generate",     type=int,   default=1)
parser.add_argument("-min",  "--min_length",     type=int,   default=5)
parser.add_argument("-max",  "--max_length",     type=int,   default=50)
parser.add_argument("-task", "--task_label",     type=str,   default="AIP")
parser.add_argument("-temp", "--temperature",    type=float, default=0.9)
parser.add_argument("-tk",   "--top_k",          type=int,   default=950)
parser.add_argument("-tp",   "--top_p",          type=float, default=0.95)
parser.add_argument("-tok",  "--max_new_tokens", type=int,   default=None)
args = parser.parse_args()

if args.max_new_tokens is None:
    args.max_new_tokens = math.ceil(args.max_length * 1.3)


# ── quality gate thresholds ───────────────────────────────────────────────────
# these match the values described in the manuscript methods section

CANONICAL_AA       = set("ACDEFGHIKLMNPQRSTVWY")
MAX_SINGLE_AA_FRAC = 0.40   # gate 3: any residue > 40% composition
MIN_UNIQUE_AA      = 4      # gate 4: at least 4 distinct residues
AA_MAX_FREQ = {             # gate 5: synthesis-limiting residues
    "C": 0.20,
    "W": 0.15,
    "M": 0.15,
}

MODEL_NAME = "nferruz/ProtGPT2"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── load model ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("device: %s | length: %d-%d aa | n per prompt: %d",
         device, args.min_length, args.max_length, args.n_generate)

log.info("loading ProtGPT2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
log.info("model ready\n")


# ── quality gates ─────────────────────────────────────────────────────────────

def passes_gates(seq, prompt, seen):
    """
    Five sequential quality gates from the manuscript (Section 3.3).
    Returns (True, 'ok') or (False, rejection_reason).
    """
    # gate 1: canonical amino acids only
    if not all(c in CANONICAL_AA for c in seq):
        return False, "non_canonical"

    # gate 2: length
    if not (args.min_length <= len(seq) <= args.max_length):
        return False, "length"

    counts = Counter(seq)

    # gate 3: single residue dominance
    if counts.most_common(1)[0][1] / len(seq) > MAX_SINGLE_AA_FRAC:
        return False, "repetitive"

    # gate 4: minimum diversity
    if len(set(seq)) < MIN_UNIQUE_AA:
        return False, "low_diversity"

    # gate 5: synthesis-limiting residue composition
    for aa, thresh in AA_MAX_FREQ.items():
        if counts.get(aa, 0) / len(seq) > thresh:
            return False, f"excess_{aa}"

    # duplicate check against already-accepted sequences this session
    if seq in seen:
        return False, "duplicate"

    return True, "ok"


# ── fasta reader ──────────────────────────────────────────────────────────────

def read_fasta(path):
    entries, header, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and parts:
                    entries.append((header, "".join(parts)))
                header, parts = line[1:].strip(), []
            elif line:
                parts.append(line.upper())
    if header and parts:
        entries.append((header, "".join(parts)))
    return entries


# ── generation ────────────────────────────────────────────────────────────────

def generate_from_prompt(prompt_seq, n, label):
    """
    Generate n valid continuations from prompt_seq.
    Tries up to n*40 times before giving up on remaining slots.
    """
    seen      = set()
    accepted  = []
    rejected  = Counter()
    inputs    = tokenizer(prompt_seq, return_tensors="pt").to(device)
    max_tries = n * 40

    pbar = tqdm(total=n, desc=f"  {label}", unit="seq", leave=False)

    for _ in range(max_tries):
        if len(accepted) >= n:
            break

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample            = True,
                top_k                = args.top_k,
                top_p                = args.top_p,
                temperature          = args.temperature,
                max_new_tokens       = args.max_new_tokens,
                num_return_sequences = 1,
                pad_token_id         = tokenizer.eos_token_id,
            )

        # decode only the newly generated tokens
        new_ids  = out[0][inputs["input_ids"].shape[1]:]
        raw      = tokenizer.decode(new_ids, skip_special_tokens=True)
        seq      = "".join(c for c in raw.upper() if c in CANONICAL_AA)

        ok, reason = passes_gates(seq, prompt_seq, seen)
        if ok:
            seen.add(seq)
            accepted.append(seq)
            pbar.update(1)
        else:
            rejected[reason] += 1

    pbar.close()

    if len(accepted) < n:
        log.warning("%s: only %d/%d accepted | rejections: %s",
                    label, len(accepted), n, dict(rejected))

    return accepted


# ── main ──────────────────────────────────────────────────────────────────────

in_dir  = Path(args.input_folder)
out_dir = Path(args.output_folder)

if not in_dir.exists():
    raise FileNotFoundError(f"input folder not found: {in_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

combined_fasta = out_dir / f"{args.task_label}_synthetic.fasta"
log_csv        = out_dir / f"{args.task_label}_generation_log.csv"

fasta_files = sorted(
    f for f in in_dir.iterdir()
    if f.suffix.lower() in {".fasta", ".fa", ".txt"}
)
log.info("found %d input files in %s", len(fasta_files), in_dir)

n_generated = 0
n_skipped   = 0

log_fields = ["seq_id", "source_file", "prompt", "generated_seq",
              "length", "temperature", "top_k", "top_p", "task"]

with open(combined_fasta, "w") as fasta_out, \
     open(log_csv, "w", newline="") as csv_out:

    writer = csv.DictWriter(csv_out, fieldnames=log_fields)
    writer.writeheader()

    for fasta_file in tqdm(fasta_files, desc="prompts", unit="file"):
        entries = read_fasta(fasta_file)
        if not entries:
            n_skipped += 1
            continue

        for idx, (header, prompt) in enumerate(entries, 1):
            # skip degenerate inputs
            if len(prompt) < 3 or not all(c in CANONICAL_AA for c in prompt):
                n_skipped += 1
                continue

            label    = f"{fasta_file.stem}_{idx}"
            accepted = generate_from_prompt(prompt, args.n_generate, label)

            for gi, seq in enumerate(accepted, 1):
                seq_id = f"{args.task_label}_syn_{fasta_file.stem}_{idx}_g{gi}"

                # individual file (needed by embedding pipeline)
                with open(out_dir / f"{seq_id}.fasta", "w") as f:
                    f.write(f">{seq_id}\n{seq}\n")

                fasta_out.write(f">{seq_id}\n{seq}\n")
                fasta_out.flush()

                writer.writerow({
                    "seq_id"       : seq_id,
                    "source_file"  : fasta_file.name,
                    "prompt"       : prompt,
                    "generated_seq": seq,
                    "length"       : len(seq),
                    "temperature"  : args.temperature,
                    "top_k"        : args.top_k,
                    "top_p"        : args.top_p,
                    "task"         : args.task_label,
                })
                n_generated += 1

log.info("done — %d sequences written to %s", n_generated, out_dir)
log.info("skipped inputs: %d", n_skipped)
log.info("combined fasta: %s", combined_fasta)
log.info("log: %s", log_csv)
log.info("")
log.info("next: cd-hit -i pos_train.fasta -i2 %s -o clean_synthetic.fasta -c 0.40 -n 2",
         combined_fasta)
