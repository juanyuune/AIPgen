import gc
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


DATALABEL = "AIP"

def data_label():
    return DATALABEL


# ── paths ─────────────────────────────────────────────────────────────────────
# switch between raw ESM-2 (stages 1/2) and KG-fused (stages 3/4)
# by commenting/uncommenting the relevant block inside each function

BASE_ESM2 = "C:/jupyter/juan/AIP/dataset/esm2"
BASE_KG   = "C:/jupyter/juan/AIP/dataset/kg_fused"


# ── independent mode (no synthetic) ──────────────────────────────────────────
# stage 1: raw ESM-2 | stage 3: KG-fused

def MCNN_data_load(DATA_TYPE, MAXSEQ):

    # stage 1 — raw ESM-2
    pos_train = f"{BASE_ESM2}/pos_train.npy"
    neg_train = f"{BASE_ESM2}/neg_train.npy"
    pos_test  = f"{BASE_ESM2}/pos_test.npy"
    neg_test  = f"{BASE_ESM2}/neg_test.npy"

    # stage 3 — KG-fused (uncomment to switch)
    # pos_train = f"{BASE_KG}/pos_real_train_kg.npy"
    # neg_train = f"{BASE_KG}/neg_real_train_kg.npy"
    # pos_test  = f"{BASE_KG}/pos_real_test_kg.npy"
    # neg_test  = f"{BASE_KG}/neg_real_test_kg.npy"

    x_train, y_train = _load_split(pos_train, neg_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    x_test, y_test = _load_split(pos_test, neg_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    return x_train, y_train, x_test, y_test


# ── cross-validation mode (with synthetic) ───────────────────────────────────
# stage 2: raw ESM-2 + synthetic | stage 4: KG-fused + synthetic
# synthetic sequences are returned separately and added per fold in training only

def MCNN_data_load_with_synthetic(DATA_TYPE, MAXSEQ):

    # stage 2 — raw ESM-2 + synthetic
    pos_train = f"{BASE_ESM2}/pos_train.npy"
    neg_train = f"{BASE_ESM2}/neg_train.npy"
    syn_train = f"{BASE_ESM2}/synthetic_train.npy"
    pos_test  = f"{BASE_ESM2}/pos_test.npy"
    neg_test  = f"{BASE_ESM2}/neg_test.npy"

    # stage 4 — KG-fused + synthetic (uncomment to switch)
    # pos_train = f"{BASE_KG}/pos_real_train_kg.npy"
    # neg_train = f"{BASE_KG}/neg_real_train_kg.npy"
    # syn_train = f"{BASE_KG}/pos_synthetic_train_kg.npy"
    # pos_test  = f"{BASE_KG}/pos_real_test_kg.npy"
    # neg_test  = f"{BASE_KG}/neg_real_test_kg.npy"

    # real training data
    x_pos = np.load(pos_train).astype(np.float32)
    x_neg = np.load(neg_train).astype(np.float32)
    y_pos = np.ones(x_pos.shape[0],  dtype=int)
    y_neg = np.zeros(x_neg.shape[0], dtype=int)

    x_train = np.concatenate([x_pos, x_neg], axis=0)
    y_train = np.concatenate([y_pos, y_neg], axis=0)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    x_synthetic = np.load(syn_train).astype(np.float32)

    # test set is real sequences only — synthetic never touches evaluation
    x_test, y_test = _load_split(pos_test, neg_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    print(f"train real : {x_train.shape}  "
          f"pos={y_train.sum()}  neg={(y_train==0).sum()}")
    print(f"synthetic  : {x_synthetic.shape}")
    print(f"test       : {x_test.shape}  pos={y_test[:,1].sum():.0f}")

    if x_train.shape[1:] != x_synthetic.shape[1:]:
        raise ValueError(
            f"shape mismatch — real {x_train.shape[1:]} "
            f"vs synthetic {x_synthetic.shape[1:]}. "
            f"Re-run embedding on synthetic with same MAXSEQ and PLM."
        )

    del x_pos, x_neg
    gc.collect()

    return x_train, y_train, x_synthetic, x_test, y_test


# ── shared loader ─────────────────────────────────────────────────────────────

def _load_split(pos_path, neg_path):
    x_pos = np.load(pos_path).astype(np.float32)
    x_neg = np.load(neg_path).astype(np.float32)

    y_pos = np.ones(x_pos.shape[0],  dtype=int)
    y_neg = np.zeros(x_neg.shape[0], dtype=int)

    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    y = tf.keras.utils.to_categorical(y, 2)

    gc.collect()
    return x, y
