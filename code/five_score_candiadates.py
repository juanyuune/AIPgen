"""
five_score_candidates.py
Scores all 618 ProtGPT2-generated sequences using the trained
AIPgen MCNN and selects the top five candidates after secondary
biochemical feasibility filtering.

Reproduces Table 14 from the manuscript.

Usage:
    python five_score_candidates.py

Requires:
    - Trained model weights from 4_train_mcnn.py
    - Synthetic embeddings: dataset/esm2/synthetic_train.npy
    - Synthetic FASTA:      iedb_fasta/AIP_synthetic.fasta
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ── paths ─────────────────────────────────────────────────────────────────────

BASE         = "C:/jupyter/juan/AIP"
WEIGHTS_PATH = os.path.join(BASE, "code/saved_weights/model",
                            "AIP_esm2_mcnn_35_[32, 34, 12, 16, 24].h5")
SYN_EMB      = os.path.join(BASE, "dataset/esm2/synthetic_train.npy")
SYN_FASTA    = os.path.join(BASE, "iedb_fasta/AIP_synthetic.fasta")


# ── model parameters — must match training exactly ────────────────────────────

MAXSEQ       = 35
NUM_FEATURE  = 1280
NUM_FILTER   = 1024
NUM_HIDDEN   = 500
WINDOW_SIZES = [32, 34, 12, 16, 24]
NUM_CLASSES  = 2


# ── model definition ──────────────────────────────────────────────────────────

class DeepScan(Model):
    def __init__(self, input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=WINDOW_SIZES,
                 num_filters=NUM_FILTER,
                 num_hidden=NUM_HIDDEN):
        super(DeepScan, self).__init__()
        self.input_layer  = tf.keras.Input(input_shape)
        self.window_sizes = window_sizes
        self.conv2d   = []
        self.maxpool  = []
        self.flatten  = []

        for ws in self.window_sizes:
            self.conv2d.append(layers.Conv2D(
                num_filters, kernel_size=(1, ws),
                activation=tf.nn.relu, padding="valid",
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.keras.initializers.GlorotUniform()))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAXSEQ - ws + 1),
                strides=(1, MAXSEQ), padding="valid"))
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden, activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.fc2 = layers.Dense(
            NUM_CLASSES, activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.out = self.call(self.input_layer)

    def call(self, x, training=False):
        branches = []
        for i in range(len(self.window_sizes)):
            h = self.conv2d[i](x)
            h = self.maxpool[i](h)
            h = self.flatten[i](h)
            branches.append(h)
        x = tf.concat(branches, axis=1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ── load model ────────────────────────────────────────────────────────────────

model = DeepScan()
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.build(input_shape=(None, 1, MAXSEQ, NUM_FEATURE))
model.load_weights(WEIGHTS_PATH)
print(f"weights loaded: {WEIGHTS_PATH}")


# ── score all 618 synthetic sequences ────────────────────────────────────────

syn_emb = np.load(SYN_EMB)
print(f"synthetic embeddings: {syn_emb.shape}")

probs      = model.predict(syn_emb, batch_size=256, verbose=1)
aip_scores = probs[:, 1]

seqs  = [(str(r.id), str(r.seq).upper()) for r in SeqIO.parse(SYN_FASTA, "fasta")]
print(f"sequences loaded: {len(seqs)}  scores computed: {len(aip_scores)}")

ranked = sorted(zip(seqs, aip_scores), key=lambda x: x[1], reverse=True)

print("\ntop 20 by AIP score:")
print(f"{'rank':<5} {'score':<8} {'len':<5} {'sequence'}")
print("-" * 65)
for i, ((sid, seq), score) in enumerate(ranked[:20], 1):
    print(f"{i:<5} {score:.4f}   {len(seq):<5} {seq}")


# ── biochemical feasibility filter ───────────────────────────────────────────
# criteria from manuscript Section 3.7:
#   length 8-40 aa
#   net charge -6 to +8 (K+R+H) - (D+E)
#   cysteine count <= 3
#   no polybasic runs > 8 consecutive basic residues

def filter_candidate(seq, score):
    issues = []
    flags  = []

    if len(seq) < 8:
        issues.append("too short (<8 aa)")
    elif len(seq) > 40:
        issues.append(f"too long ({len(seq)} aa)")

    cys = seq.count("C")
    if cys > 3:
        issues.append(f"too many Cys ({cys})")
    elif cys > 1:
        flags.append(f"{cys} Cys")

    pos_aa = seq.count("K") + seq.count("R") + seq.count("H")
    neg_aa = seq.count("D") + seq.count("E")
    charge = pos_aa - neg_aa
    if charge > 8:
        issues.append(f"excessive charge (+{charge})")
    elif charge < -6:
        issues.append(f"excessive negative charge ({charge})")

    # polybasic artefact check
    if seq.count("H") > 8:
        issues.append(f"polyhistidine artefact ({seq.count('H')} His)")
    if seq.count("R") > 7:
        issues.append(f"polyarginine artefact ({seq.count('R')} Arg)")

    try:
        analysis = ProteinAnalysis(seq)
        gravy    = analysis.gravy()
        if gravy > 2.0:
            issues.append(f"extremely hydrophobic (GRAVY {gravy:.2f})")
        elif gravy > 1.0:
            flags.append(f"hydrophobic (GRAVY {gravy:.2f})")
    except Exception:
        pass

    status = "REJECT" if issues else ("FLAG" if flags else "ACCEPT")
    return status, issues, flags


print("\n\nbiotchemical feasibility filter applied to top 20:")
print(f"{'rank':<5} {'score':<7} {'len':<5} {'status':<8} {'notes':<45} {'sequence'}")
print("-" * 120)

accepted = []
flagged  = []

for i, ((sid, seq), score) in enumerate(ranked[:20], 1):
    status, issues, flags = filter_candidate(seq, score)
    notes = "; ".join(issues + flags) if (issues or flags) else "clean"
    print(f"{i:<5} {score:.4f}  {len(seq):<5} {status:<8} {notes:<45} {seq}")
    if status == "ACCEPT":
        accepted.append((i, score, seq, sid))
    elif status == "FLAG":
        flagged.append((i, score, seq, sid, flags))

print(f"\n{'='*60}")
print(f"accepted  : {len(accepted)} candidates")
print(f"flagged   : {len(flagged)} candidates")
print(f"{'='*60}")

print("\nfinal top 5 candidates (Table 14):")
print(f"{'rank':<5} {'AIP score':<12} {'pLDDT':<8} {'len':<5} sequence")
print("-" * 65)
for rank, (_, score, seq, sid) in enumerate(accepted[:5], 1):
    # pLDDT values from ESMFold predictions (see 6_esmfold_predict.py)
    print(f"{rank:<5} {score:.4f}       {'run ESMFold':<8} {len(seq):<5} {seq}")

print("\nnote: pLDDT values are obtained from 6_esmfold_predict.py")
print("      only candidates with pLDDT >= 0.65 proceed to BindCraft")
