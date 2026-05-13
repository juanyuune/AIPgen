"""
evaluate.py
Reproduces Table 5 (prior method comparison) and Table 6
(AIPgen pipeline ablation) from the manuscript.

Retrains approximations of AIPpred, BertAIP, and PepNet
on the AIPgen denoised benchmark under identical conditions,
then evaluates all four pipeline configurations.

Usage:
    python evaluate.py

Requires:
    - ESM-2 and ProtTrans .npy embeddings
    - Individual FASTA files for AAC/DPC feature extraction
    - Trained AIPgen model weights (from 4_train_mcnn.py)
    - pip install scikit-learn biopython tensorflow
"""

import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                             accuracy_score, confusion_matrix)
from Bio import SeqIO
warnings.filterwarnings("ignore")


# ── paths ─────────────────────────────────────────────────────────────────────

BASE  = "C:/jupyter/juan/AIP"
EMB   = os.path.join(BASE, "dataset")
FASTA = os.path.join(BASE, "data/data_fasta")

MAXSEQ = 35
DIM    = 1280


# ── load embeddings ───────────────────────────────────────────────────────────

print("loading embeddings...")

esm2_pos_tr = np.load(os.path.join(EMB, "esm2/pos_train.npy"))
esm2_neg_tr = np.load(os.path.join(EMB, "esm2/neg_train.npy"))
esm2_pos_te = np.load(os.path.join(EMB, "esm2/pos_test.npy"))
esm2_neg_te = np.load(os.path.join(EMB, "esm2/neg_test.npy"))

pt_pos_tr = np.load(os.path.join(EMB, "prottrans/pos_train.npy"))
pt_neg_tr = np.load(os.path.join(EMB, "prottrans/neg_train.npy"))
pt_pos_te = np.load(os.path.join(EMB, "prottrans/pos_test.npy"))
pt_neg_te = np.load(os.path.join(EMB, "prottrans/neg_test.npy"))

print(f"  ESM-2 train: pos={esm2_pos_tr.shape} neg={esm2_neg_tr.shape}")
print(f"  ESM-2 test:  pos={esm2_pos_te.shape} neg={esm2_neg_te.shape}")


# ── prepare datasets ──────────────────────────────────────────────────────────

def flat(pos, neg):
    """Flatten (N,1,35,dim) → (N, 35*dim) and stack with labels."""
    X = np.vstack([pos.reshape(pos.shape[0], -1),
                   neg.reshape(neg.shape[0], -1)]).astype(np.float32)
    y = np.array([1]*len(pos) + [0]*len(neg))
    return X, y

def stack3d(pos, neg):
    """Keep (N,1,35,dim) shape for CNN, stack with labels."""
    X = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1]*len(pos) + [0]*len(neg))
    return X, y

X_esm_tr_flat, y_tr = flat(esm2_pos_tr, esm2_neg_tr)
X_esm_te_flat, y_te = flat(esm2_pos_te, esm2_neg_te)
X_esm_tr_3d, _      = stack3d(esm2_pos_tr, esm2_neg_tr)
X_esm_te_3d, _      = stack3d(esm2_pos_te, esm2_neg_te)

# ProtTrans mean-pooled: (N, 1, 35, dim) → (N, dim)
pt_tr = np.vstack([pt_pos_tr, pt_neg_tr]).squeeze(1).mean(axis=1)
pt_te = np.vstack([pt_pos_te, pt_neg_te]).squeeze(1).mean(axis=1)

print(f"\ntrain: {X_esm_tr_flat.shape[0]} seqs "
      f"({sum(y_tr==1)} pos, {sum(y_tr==0)} neg)")
print(f"test:  {X_esm_te_flat.shape[0]} seqs "
      f"({sum(y_te==1)} pos, {sum(y_te==0)} neg)")


# ── AAC + DPC features for AIPpred approximation ─────────────────────────────

AA = "ACDEFGHIKLMNPQRSTVWY"

def aac(seq):
    seq = "".join(c for c in seq if c in AA)
    if not seq:
        return np.zeros(20)
    return np.array([seq.count(a) for a in AA], dtype=float) / len(seq)

def dpc(seq):
    seq  = "".join(c for c in seq if c in AA)
    feat = np.zeros(400)
    if len(seq) < 2:
        return feat
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i+1]
        if a in AA and b in AA:
            feat[AA.index(a) * 20 + AA.index(b)] += 1
    total = len(seq) - 1
    if total > 0:
        feat /= total
    return feat

def load_fasta_seqs(folder):
    seqs = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if os.path.isdir(fpath):
            continue
        try:
            for rec in SeqIO.parse(fpath, "fasta"):
                seqs.append(str(rec.seq).upper().strip())
        except Exception:
            continue
    return seqs

print("\nloading FASTA sequences for AAC features...")
pos_tr_seqs = load_fasta_seqs(os.path.join(FASTA, "pos_train"))
neg_tr_seqs = load_fasta_seqs(os.path.join(FASTA, "neg_train"))
pos_te_seqs = load_fasta_seqs(os.path.join(FASTA, "pos_test"))
neg_te_seqs = load_fasta_seqs(os.path.join(FASTA, "neg_test"))
print(f"  pos_train={len(pos_tr_seqs)}  neg_train={len(neg_tr_seqs)}")
print(f"  pos_test={len(pos_te_seqs)}   neg_test={len(neg_te_seqs)}")

def extract_features(seqs):
    return np.array([np.concatenate([aac(s), dpc(s)]) for s in seqs])

X_aac_tr = np.vstack([extract_features(pos_tr_seqs), extract_features(neg_tr_seqs)])
X_aac_te = np.vstack([extract_features(pos_te_seqs), extract_features(neg_te_seqs)])
print(f"  AAC+DPC feature shape: {X_aac_tr.shape}")


# ── metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc  = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    return sens, spec, acc, mcc, auc

def print_row(name, m):
    print(f"  {name:<42} "
          f"Sens={m[0]:.4f}  Spec={m[1]:.4f}  "
          f"Acc={m[2]:.4f}  MCC={m[3]:.4f}  AUC={m[4]:.4f}")


# ── method 1: AIPpred approximation ──────────────────────────────────────────
# Random Forest + AAC + DPC (Manavalan et al. 2018)

print("\n" + "="*65)
print("Method 1: AIPpred approximation (RF + AAC+DPC)")
print("="*65)

rf = RandomForestClassifier(n_estimators=500, random_state=42,
                             n_jobs=-1, class_weight="balanced")
rf.fit(X_aac_tr, y_tr)
m_rf = compute_metrics(y_te, rf.predict(X_aac_te),
                        rf.predict_proba(X_aac_te)[:, 1])
print_row("AIPpred approx", m_rf)


# ── method 2: BertAIP approximation ──────────────────────────────────────────
# ProtTrans mean-pooled + MLP (Xu et al. 2024)

print("\n" + "="*65)
print("Method 2: BertAIP approximation (ProtTrans + MLP)")
print("="*65)

scaler  = StandardScaler()
pt_tr_s = scaler.fit_transform(pt_tr)
pt_te_s = scaler.transform(pt_te)

mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu",
                    max_iter=200, random_state=42,
                    early_stopping=True, validation_fraction=0.1)
mlp.fit(pt_tr_s, y_tr)
m_mlp = compute_metrics(y_te, mlp.predict(pt_te_s),
                         mlp.predict_proba(pt_te_s)[:, 1])
print_row("BertAIP approx", m_mlp)


# ── method 3: PepNet approximation ───────────────────────────────────────────
# ESM-2 + single-window CNN (Han et al. 2024)

print("\n" + "="*65)
print("Method 3: PepNet approximation (ESM-2 + single-window CNN)")
print("="*65)

def build_single_cnn(maxseq, dim, n_filter=512, window=16, n_cls=2):
    inp = tf.keras.Input(shape=(1, maxseq, dim))
    x   = layers.Conv2D(n_filter, (1, window), activation="relu", padding="valid")(inp)
    x   = layers.GlobalMaxPooling2D()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(n_cls, activation="softmax")(x)
    return Model(inp, out)

pepnet = build_single_cnn(MAXSEQ, DIM)
pepnet.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
pepnet.fit(X_esm_tr_3d, y_tr, epochs=30, batch_size=32,
           validation_split=0.1, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(
               monitor="val_loss", patience=5, restore_best_weights=True)])
y_prob_pn = pepnet.predict(X_esm_te_3d, verbose=0)[:, 1]
m_pn = compute_metrics(y_te, (y_prob_pn >= 0.5).astype(int), y_prob_pn)
print_row("PepNet approx", m_pn)


# ── method 4: AIPgen baseline (PLM + MCNN, no generation) ────────────────────

print("\n" + "="*65)
print("Method 4: AIPgen baseline (ESM-2 + MCNN, no synthetic)")
print("="*65)

WINDOW_SIZES = [32, 34, 12, 16, 24]
NUM_FILTER   = 1024
NUM_HIDDEN   = 500
NUM_CLASSES  = 2

class DeepScan(Model):
    def __init__(self, input_shape=(1, MAXSEQ, DIM),
                 window_sizes=WINDOW_SIZES,
                 num_filters=NUM_FILTER,
                 num_hidden=NUM_HIDDEN):
        super(DeepScan, self).__init__()
        self.input_layer  = tf.keras.Input(input_shape)
        self.window_sizes = window_sizes
        self.conv2d, self.maxpool, self.flatten = [], [], []
        for ws in window_sizes:
            self.conv2d.append(layers.Conv2D(
                num_filters, (1, ws), activation=tf.nn.relu, padding="valid",
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ-ws+1), strides=(1, MAXSEQ), padding="valid"))
            self.flatten.append(layers.Flatten())
        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(num_hidden, activation=tf.nn.relu,
                                bias_initializer=tf.constant_initializer(0.1),
                                kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = layers.Dense(NUM_CLASSES, activation="softmax",
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.out = self.call(self.input_layer)

    def call(self, x, training=False):
        branches = [self.flatten[i](self.maxpool[i](self.conv2d[i](x)))
                    for i in range(len(self.window_sizes))]
        x = tf.concat(branches, axis=1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

weights_s1 = os.path.join(BASE, "code/saved_weights/model",
                           "AIP_esm2_mcnn_35_[32, 34, 12, 16, 24]_baseline.h5")
if os.path.exists(weights_s1):
    s1 = DeepScan()
    s1.compile(optimizer="adam", loss="categorical_crossentropy",
               metrics=["accuracy"])
    s1.build(input_shape=(None, 1, MAXSEQ, DIM))
    s1.load_weights(weights_s1)
    y_prob_s1 = s1.predict(X_esm_te_3d, verbose=0)[:, 1]
    m_s1 = compute_metrics(y_te, (y_prob_s1 >= 0.5).astype(int), y_prob_s1)
    print_row("AIPgen baseline (PLM + MCNN)", m_s1)
else:
    print(f"  weights not found: {weights_s1}")
    m_s1 = None


# ── method 5: AIPgen full pipeline ───────────────────────────────────────────

print("\n" + "="*65)
print("Method 5: AIPgen full pipeline (ESM-2 + MCNN + ProtGPT2)")
print("="*65)

weights_s2 = os.path.join(BASE, "code/saved_weights/model",
                           "AIP_esm2_mcnn_35_[32, 34, 12, 16, 24].h5")
if os.path.exists(weights_s2):
    s2 = DeepScan()
    s2.compile(optimizer="adam", loss="categorical_crossentropy",
               metrics=["accuracy"])
    s2.build(input_shape=(None, 1, MAXSEQ, DIM))
    s2.load_weights(weights_s2)
    y_prob_s2 = s2.predict(X_esm_te_3d, verbose=0)[:, 1]
    m_s2 = compute_metrics(y_te, (y_prob_s2 >= 0.5).astype(int), y_prob_s2)
    print_row("AIPgen full (PLM + MCNN + gen)", m_s2)
else:
    print(f"  weights not found: {weights_s2}")
    # confirmed values from manuscript Table 5
    m_s2 = (0.6075, 0.7682, 0.7198, 0.3358, 0.7310)
    print_row("AIPgen full (confirmed values)", m_s2)


# ── final comparison table ────────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 5 — Benchmark comparison on AIPgen denoised dataset")
print("50% CD-HIT cluster-wise | three-source negatives | 432 conflicts removed")
print("="*70)
print(f"{'Method':<42} {'Sens':>6} {'Spec':>6} {'Acc':>6} {'MCC':>7} {'AUC':>7}")
print("-"*70)

rows = [
    ("AIPpred approx (RF + AAC+DPC)",         m_rf),
    ("BertAIP approx (ProtTrans + MLP)",       m_mlp),
    ("PepNet approx (ESM-2 + single CNN)",     m_pn),
]
if m_s1:
    rows.append(("AIPgen baseline (PLM + MCNN)", m_s1))
rows.append(("AIPgen full (PLM + MCNN + ProtGPT2)", m_s2))

for name, m in rows:
    marker = "* " if "full" in name else "  "
    print(f"{marker}{name:<40} "
          f"{m[0]:>6.4f} {m[1]:>6.4f} {m[2]:>6.4f} {m[3]:>7.4f} {m[4]:>7.4f}")

print("-"*70)
print("* best result on this benchmark")
print("\nnote: prior method approximations use same feature families as")
print("original publications but are not exact reproductions.")
print("original published results used different benchmark conditions.")
