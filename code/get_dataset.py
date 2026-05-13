"""
get_dataset.py
Stacks individual per-sequence embedding files into a single
.npy array ready for the MCNN classifier.

Handles all PLM formats used in this study:
  .prottrans  — numpy binary or plain text (legacy)
  .esm2       — numpy binary or pickle
  .ankh       — numpy binary
  .npy        — numpy binary

Output shape: (N, 1, MAXSEQ, D)
  N      = number of sequences
  1      = channel dim expected by Conv2D
  MAXSEQ = max sequence length (default 35)
  D      = embedding dimension (1024 ProtTrans, 1280 ESM-2,
           768 Ankh-base, 1536 Ankh-large)

Usage:
    python get_dataset.py \
        -in    C:/jupyter/juan/AIP/emb/esm2/pos_train \
        -out   C:/jupyter/juan/AIP/dataset/esm2/pos_train.npy \
        -dt    .esm2 \
        -maxseq 35
"""

import os
import time
import argparse
import pickle

import numpy as np
import torch
import h5py


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("-in",     "--path_input",   required=True,
                    help="folder of individual embedding files")
parser.add_argument("-out",    "--path_output",  required=True,
                    help="output .npy file path")
parser.add_argument("-dt",     "--data_type",    required=True,
                    help="file extension: .prottrans .esm2 .ankh .npy")
parser.add_argument("-maxseq", "--max_sequence", type=int, default=35,
                    help="max sequence length for padding (default: 35)")


# ── format detection ──────────────────────────────────────────────────────────

def detect_format(path):
    try:
        with open(path, "rb") as f:
            magic = f.read(16)
        if magic[:6] == b'\x93NUMPY': return "numpy"
        if magic[:4] == b'\x89HDF':   return "hdf5"
        if magic[:2] == b'PK':        return "torch"
        try:
            magic.decode("ascii")
            if chr(magic[0]) in "0123456789-. \t\n":
                return "text"
        except Exception:
            pass
        return "unknown"
    except Exception:
        return "unknown"


# ── format-specific loaders ───────────────────────────────────────────────────

def load_hdf5(path):
    with h5py.File(path, "r") as f:
        for k in f.keys():
            arr = np.array(f[k])
            if arr.ndim >= 1:
                return arr
    raise ValueError(f"no readable array in HDF5: {path}")


def load_torch(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(data, "numpy"):
        return data.numpy()
    if isinstance(data, dict):
        for key in ("mean_representations", "representations"):
            if key in data:
                layer = max(data[key].keys())
                t     = data[key][layer]
                return t.numpy() if hasattr(t, "numpy") else np.array(t)
        for v in data.values():
            if hasattr(v, "numpy"):
                return v.numpy()
    raise ValueError(f"cannot extract array from torch file: {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.array(data)


def load_file(path):
    fmt = detect_format(path)

    if fmt == "numpy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        return data
    if fmt == "hdf5":   return load_hdf5(path)
    if fmt == "torch":  return load_torch(path)
    if fmt == "text":   return np.loadtxt(path, dtype=np.float32)

    # unknown — try each loader in turn
    for loader in [lambda p: np.load(p, allow_pickle=True),
                   lambda p: np.loadtxt(p, dtype=np.float32),
                   load_torch, load_pickle, load_hdf5]:
        try:
            data = loader(path)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            return data
        except Exception:
            pass

    with open(path, "rb") as f:
        magic = f.read(32)
    raise ValueError(f"all loaders failed for {path}\nfirst 32 bytes: {magic}")


# ── shape normalisation → (L, D) ─────────────────────────────────────────────

def normalize(data):
    data = np.array(data, dtype=np.float32)
    if data.ndim == 2: return data
    if data.ndim == 3:
        if data.shape[0] == 1: return data[0]
        if data.shape[1] == 1: return data[:, 0, :]
        return data.mean(axis=0)
    if data.ndim == 1: return data[np.newaxis, :]
    raise ValueError(f"unsupported embedding shape: {data.shape}")


# ── pad or truncate to MAXSEQ ─────────────────────────────────────────────────

def pad(x, maxseq):
    L, D  = x.shape
    out   = np.zeros((maxseq, D), dtype=np.float32)
    out[:min(L, maxseq), :] = x[:min(L, maxseq), :]
    return out


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.path_input)
                   if f.endswith(args.data_type))

    if not files:
        raise ValueError(f"no {args.data_type} files in {args.path_input}")

    print(f"found {len(files)} files  |  maxseq={args.max_sequence}  |  "
          f"output: {args.path_output}")

    data_list = []
    skipped   = []
    emb_dim   = None
    t0        = time.time()

    for i, fname in enumerate(files, 1):
        path = os.path.join(args.path_input, fname)
        try:
            emb = pad(normalize(load_file(path)), args.max_sequence)
            emb = emb[np.newaxis, :, :]   # (1, MAXSEQ, D)
            data_list.append(emb)
            if emb_dim is None:
                emb_dim = emb.shape[-1]
                print(f"embedding dim: {emb_dim}  sample shape: {emb.shape}")
        except Exception as e:
            print(f"skip {fname}: {e}")
            skipped.append(fname)

        if i % 200 == 0:
            print(f"  {i}/{len(files)}  elapsed {time.time()-t0:.0f}s")

    if not data_list:
        raise RuntimeError(f"no valid embeddings loaded from {args.path_input}")

    X = np.concatenate(data_list, axis=0)
    X = X.reshape(X.shape[0], 1, args.max_sequence, X.shape[-1])

    os.makedirs(os.path.dirname(os.path.abspath(args.path_output)), exist_ok=True)
    np.save(args.path_output, X)

    print(f"\nsaved {args.path_output}")
    print(f"shape   : {X.shape}")
    print(f"loaded  : {len(data_list)}  skipped: {len(skipped)}  "
          f"time: {time.time()-t0:.1f}s")

    if skipped:
        print(f"\nskipped files:")
        for s in skipped[:10]:
            print(f"  {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")
