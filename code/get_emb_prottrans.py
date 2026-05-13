"""
get_emb_prottrans.py
ProtT5-XL per-residue embeddings for AIP sequences.

Reads individual FASTA files, generates ProtT5-XL encoder
representations, and saves each as a .prottrans numpy file.

Usage:
    python get_emb_prottrans.py \
        -in  C:/jupyter/juan/AIP/data/data_fasta/pos_train \
        -out C:/jupyter/juan/AIP/data/prottrans_embeddings/pos_train

Requirements:
    pip install transformers torch sentencepiece
"""

import os
import gc
import time
import logging
import argparse

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("-in",  "--path_input",  required=True,
                    help="folder of individual .fasta files")
parser.add_argument("-out", "--path_output", required=True,
                    help="folder to save .prottrans embedding files")


# ── model ─────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    log.info("loading ProtT5-XL...")
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc")
    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = model.to(device).eval()
    log.info("model ready on %s", device)
    return model, tokenizer

model, tokenizer = load_model()


# ── helpers ───────────────────────────────────────────────────────────────────

def read_fasta(path):
    seq    = ""
    seq_id = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq += line.strip()
    return [(seq_id, seq)]


def get_embedding(model, tokenizer, seqs,
                  max_residues=4000, max_seq_len=1000, max_batch=100):
    """
    Batched ProtT5 embedding — sorts by length descending to
    minimise padding waste, flushes when batch would exceed
    max_residues or max_batch.
    Returns {seq_id: np.array of shape (L, 1024)}.
    """
    results  = {}
    seq_dict = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
    batch    = []

    for seq_idx, (seq_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        seq_sp  = " ".join(list(seq))   # ProtT5 expects space-separated AA
        batch.append((seq_id, seq_sp, seq_len))

        n_res = sum(s for _, _, s in batch) + seq_len
        flush = (len(batch) >= max_batch or
                 n_res       >= max_residues or
                 seq_idx     == len(seq_dict) or
                 seq_len     >  max_seq_len)

        if not flush:
            continue

        ids, seqs_sp, lens = zip(*batch)
        batch = []

        enc      = tokenizer.batch_encode_plus(seqs_sp,
                                               add_special_tokens=True,
                                               padding="longest")
        inp_ids  = torch.tensor(enc["input_ids"]).to(device)
        attn     = torch.tensor(enc["attention_mask"]).to(device)

        try:
            with torch.no_grad():
                out = model(inp_ids, attention_mask=attn)
        except RuntimeError as e:
            log.error("RuntimeError — skipping batch: %s", e)
            continue

        for i, sid in enumerate(ids):
            L   = lens[i]
            emb = out.last_hidden_state[i, :L].detach().cpu().numpy().squeeze()
            results[sid] = emb

    return results


def log_failed(in_path, out_path):
    with open("NO_OK.txt", "a") as f:
        f.write(f"{in_path} > {out_path}\n")


def process(fasta_path, out_path):
    try:
        fname = os.path.splitext(os.path.basename(fasta_path))[0]
        seqs  = read_fasta(fasta_path)
        embs  = get_embedding(model, tokenizer, seqs)
        np.save(out_path, embs[fname])
        log.info("saved  %s  shape=%s", os.path.basename(out_path),
                 embs[fname].shape)
    except Exception as e:
        log.error("failed %s: %s", fasta_path, e)
        log_failed(fasta_path, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.path_input):
        raise FileNotFoundError(f"input not found: {args.path_input}")
    os.makedirs(args.path_output, exist_ok=True)

    fasta_files = [f for f in os.listdir(args.path_input)
                   if f.endswith(".fasta")]
    log.info("found %d fasta files in %s", len(fasta_files), args.path_input)

    for i, fname in enumerate(fasta_files, 1):
        in_path  = os.path.join(args.path_input,  fname)
        out_path = os.path.join(args.path_output,
                                os.path.splitext(fname)[0] + ".prottrans")
        if i % 100 == 0 or i == 1:
            log.info("[%d/%d] %s", i, len(fasta_files), fname)
        process(in_path, out_path)
        gc.collect()

    log.info("done — %d files embedded", len(fasta_files))
