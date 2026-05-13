"""
get_emb_esm2.py
ESM-2 650M per-residue embeddings for AIP sequences.

Reads individual FASTA files from input folder, generates
ESM-2 layer-33 representations, and saves each as a .esm2
pickle file. Pads to MAXSEQ=35 in the downstream data loader.

Usage:
    python 2_embed_esm2.py \
        -in  C:/jupyter/juan/AIP/data/data_fasta/pos_train \
        -out C:/jupyter/juan/AIP/data/esm2_embeddings/pos_train

Requirements:
    pip install fair-esm torch
"""

import os
import gc
import pickle
import logging
import argparse

import numpy as np
import torch
import esm


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("-in",  "--path_input",  required=True,
                    help="folder of individual .fasta files")
parser.add_argument("-out", "--path_output", required=True,
                    help="folder to save .esm2 embedding files")


# ── model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    log.info("loading ESM-2 650M...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    log.info("model ready on %s", device)
    return model, batch_converter

model, batch_converter = load_model()


# ── helpers ───────────────────────────────────────────────────────────────────

def read_fasta(path):
    seq = ""
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    seq_id = os.path.splitext(os.path.basename(path))[0]
    return [(seq_id, seq)]


def get_embedding(seqs):
    """
    Returns dict {seq_id: np.array of shape (L, 1280)}.
    Uses layer 33 representations, strips CLS/EOS tokens.
    """
    _, _, tokens = batch_converter(seqs)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[33], return_contacts=False)

    reps = out["representations"][33]

    results = {}
    for i, (seq_id, seq) in enumerate(seqs):
        L   = len(seq)
        emb = reps[i, 1:L+1].detach().cpu().numpy()  # drop CLS token at pos 0
        results[seq_id] = emb

    del tokens, reps, out
    torch.cuda.empty_cache()
    gc.collect()
    return results


def save_embedding(emb, path):
    with open(path, "wb") as f:
        pickle.dump(emb, f)


def log_failed(in_path, out_path):
    with open("NO_OK.txt", "a") as f:
        f.write(f"{in_path} > {out_path}\n")


def process(fasta_path, out_path):
    try:
        fname = os.path.splitext(os.path.basename(fasta_path))[0]
        seqs  = read_fasta(fasta_path)
        embs  = get_embedding(seqs)
        save_embedding(embs[fname], out_path)
        log.info("saved  %s", os.path.basename(out_path))
    except Exception as e:
        log.error("failed %s: %s", fasta_path, e)
        log_failed(fasta_path, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"input not found: {args.path_input}")
    os.makedirs(args.path_output, exist_ok=True)

    fasta_files = [f for f in os.listdir(args.path_input) if f.endswith(".fasta")]
    log.info("found %d fasta files in %s", len(fasta_files), args.path_input)

    for i, fname in enumerate(fasta_files, 1):
        in_path  = os.path.join(args.path_input,  fname)
        out_path = os.path.join(args.path_output,
                                os.path.splitext(fname)[0] + ".esm2")
        if i % 100 == 0 or i == 1:
            log.info("[%d/%d] %s", i, len(fasta_files), fname)
        process(in_path, out_path)
        gc.collect()

    log.info("done — %d files embedded", len(fasta_files))
