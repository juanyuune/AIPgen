import h5py
import os
import pickle
import math
import csv
import datetime
import time
import gc

from tqdm import tqdm
from time import gmtime, strftime

import numpy as np

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle as sk_shuffle

import tensorflow as tf
from tensorflow.keras import layers, Model

import import_data_esm2_KG as load_data


# ── configuration ─────────────────────────────────────────────────────────────

DATA_LABEL = load_data.data_label()
DATA_TYPE  = "ESM-2"   # options: BinaryMatrix, MMseqs2, ProtTrans, ESM-2, TAPE, ankh

MAXSEQ      = 35
NUM_FEATURE = 1280     # ESM-2 650M embedding dim

NUM_FILTER   = 1024
NUM_HIDDEN   = 500
BATCH_SIZE   = 256
WINDOW_SIZES = [32, 34, 12, 16, 24]

NUM_CLASSES = 2
CLASS_NAMES = ['Negative', 'Positive']

EPOCHS          = 20
K_Fold          = 5
VALIDATION_MODE = "cross"   # "cross" or "independent"
IMBALANCE       = "None"    # None, SMOTE, ADASYN, RANDOM


# ── logging ───────────────────────────────────────────────────────────────────

write_data = []
a = datetime.datetime.now()
write_data.extend([
    time.ctime(), DATA_LABEL, DATA_TYPE,
    BATCH_SIZE, NUM_HIDDEN, WINDOW_SIZES, NUM_FILTER,
    VALIDATION_MODE, IMBALANCE
])

def time_log(msg):
    print(msg, ":", strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# ── save ROC curve data ────────────────────────────────────────────────────────

def save_roc(fpr, tpr, auc):
    folder = "./PKL/AIP/feature/"
    os.makedirs(folder, exist_ok=True)
    fname = f"AIP_esm2_mcnn_{int(time.time())}.pkl"
    fpath = os.path.join(folder, fname)
    with open(fpath, "wb") as f:
        pickle.dump({"fpr": fpr, "tpr": tpr, "AUC": auc}, f)
    print(f"ROC saved: {os.path.abspath(fpath)}")


# ── data generator ────────────────────────────────────────────────────────────

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data       = data
        self.labels     = labels
        self.batch_size = batch_size
        self.indexes    = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        idx          = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data   = np.array([self.data[i]   for i in idx])
        batch_labels = np.array([self.labels[i] for i in idx])
        return batch_data, batch_labels


# ── MCNN model ────────────────────────────────────────────────────────────────
# parallel Conv2D branches over different kernel widths, then concatenate
# global max-pool collapses the spatial dimension before the FC layers

class DeepScan(Model):
    def __init__(self,
                 input_shape=(1, MAXSEQ, NUM_FEATURE),
                 window_sizes=[2, 4, 6],
                 num_filters=256,
                 num_hidden=500):
        super(DeepScan, self).__init__()
        self.input_layer  = tf.keras.Input(input_shape)
        self.window_sizes = window_sizes
        self.conv2d   = []
        self.maxpool  = []
        self.flatten  = []

        for ws in self.window_sizes:
            self.conv2d.append(
                layers.Conv2D(
                    filters=num_filters,
                    kernel_size=(1, ws),
                    activation=tf.nn.relu,
                    padding='valid',
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                )
            )
            self.maxpool.append(
                layers.MaxPooling2D(
                    pool_size=(1, MAXSEQ - ws + 1),
                    strides=(1, MAXSEQ),
                    padding='valid'
                )
            )
            self.flatten.append(layers.Flatten())

        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden,
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.fc2 = layers.Dense(
            NUM_CLASSES,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )
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


# ── imbalance handling ────────────────────────────────────────────────────────

def handle_imbalance(method, x_train, y_train):
    if method == "None":
        return x_train, y_train

    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

    x_2d = x_train.reshape(x_train.shape[0], -1)
    print(f"Before resampling: {x_2d.shape}, {y_train.shape}")

    sampler_map = {
        "SMOTE":  SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "RANDOM": RandomOverSampler(random_state=42),
    }
    x_res, y_res = sampler_map[method].fit_resample(x_2d, y_train)
    x_res = x_res.reshape(x_res.shape[0], 1, MAXSEQ, NUM_FEATURE)

    print(f"After resampling:  {x_res.shape}, {y_res.shape}")

    del x_2d
    gc.collect()

    y_res = tf.keras.utils.to_categorical(y_res, NUM_CLASSES)
    return x_res, y_res


# ── evaluation ────────────────────────────────────────────────────────────────

def model_test(model, x_test, y_test):
    assert y_test.ndim == 2 and y_test.shape[1] == 2, \
        f"y_test shape {y_test.shape}, expected (N, 2)"
    assert x_test.shape[0] == y_test.shape[0]

    print(x_test.shape)
    pred = model.predict(x_test)

    fpr, tpr, thresholds = roc_curve(y_test[:, 1], pred[:, 1])
    auc = metrics.auc(fpr, tpr)

    metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='MSCNN').plot()

    # pick threshold by max G-mean
    gmeans    = np.sqrt(tpr * (1 - fpr))
    ix        = np.argmax(gmeans)
    threshold = thresholds[ix]
    print(f"Best threshold={threshold:.4f}, G-Mean={gmeans[ix]:.4f}")

    y_pred = (pred[:, 1] >= threshold).astype(int)
    TN, FP, FN, TP = metrics.confusion_matrix(y_test[:, 1], y_pred).ravel()

    denom_mcc = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    Sens = TP / (TP + FN)       if (TP + FN) > 0    else 0.0
    Spec = TN / (FP + TN)       if (FP + TN) > 0    else 0.0
    Acc  = (TP + TN) / (TP + FP + TN + FN)
    MCC  = (TP*TN - FP*FN) / denom_mcc if denom_mcc > 0 else 0.0
    F1   = 2*TP / (2*TP + FP + FN)
    Prec = TP / (TP + FP)       if (TP + FP) > 0    else 0.0
    Rec  = TP / (TP + FN)       if (TP + FN) > 0    else 0.0

    print(
        f"TP={TP}, FP={FP}, TN={TN}, FN={FN}, "
        f"Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, "
        f"MCC={MCC:.4f}, AUC={auc:.4f}, "
        f"F1={F1:.4f}, Prec={Prec:.4f}, Recall={Rec:.4f}\n"
    )
    return TP, FP, TN, FN, Sens, Spec, Acc, MCC, auc


# ── load data ─────────────────────────────────────────────────────────────────

if VALIDATION_MODE == "cross":
    x_train, y_train, x_synthetic, x_test, y_test = \
        load_data.MCNN_data_load_with_synthetic(DATA_TYPE, MAXSEQ)
else:
    x_train, y_train, x_test, y_test = \
        load_data.MCNN_data_load(DATA_TYPE, MAXSEQ)

print("\nx_train :", x_train.shape, x_train.dtype)
print("y_train :", y_train.shape)
print("x_test  :", x_test.shape,  x_test.dtype)
print("y_test  :", y_test.shape)
if VALIDATION_MODE == "cross":
    print("x_synthetic:", x_synthetic.shape)

# quick sanity check
if VALIDATION_MODE == "cross":
    pos = y_train.sum()
    neg = (y_train == 0).sum()
    print(f"\ntrain pos={pos}  neg={neg}")
    print(f"train per fold = {x_train.shape[0]} real + {x_synthetic.shape[0]} synthetic")
    if x_train.shape[1:] != x_synthetic.shape[1:]:
        print(f"WARNING: shape mismatch real {x_train.shape[1:]} vs synthetic {x_synthetic.shape[1:]}")


# ── cross-validation training ─────────────────────────────────────────────────

if VALIDATION_MODE == "cross":
    time_log("Start cross-validation")

    skf     = StratifiedKFold(n_splits=K_Fold, shuffle=True, random_state=2)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train), 1):
        print(f"\nFold {fold}/{K_Fold}")

        X_tr = x_train[train_idx]
        Y_tr = y_train[train_idx]
        X_val = x_train[val_idx]
        Y_val = y_train[val_idx]

        # synthetic positives added to training fold only, never validation
        y_syn = np.ones(x_synthetic.shape[0], dtype=int)
        X_fold = np.concatenate([X_tr, x_synthetic], axis=0)
        Y_fold = np.concatenate([Y_tr, y_syn], axis=0)

        shuf = np.random.permutation(len(X_fold))
        X_fold = X_fold[shuf]
        Y_fold = Y_fold[shuf]

        print(f"train total: {X_fold.shape}  val: {X_val.shape}")

        Y_fold_oh = tf.keras.utils.to_categorical(Y_fold, NUM_CLASSES)
        Y_val_oh  = tf.keras.utils.to_categorical(Y_val,  NUM_CLASSES)

        gen   = DataGenerator(X_fold, Y_fold_oh, batch_size=BATCH_SIZE)
        model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN,
                         window_sizes=WINDOW_SIZES)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.build(input_shape=X_fold.shape)
        model.fit(
            gen,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
            verbose=1,
            shuffle=True
        )

        res = model_test(model, X_val, Y_val_oh)
        results.append(res)

        del X_fold, Y_fold, X_tr, Y_tr, X_val, Y_val, Y_fold_oh, Y_val_oh
        gc.collect()

    mean_r = np.mean(results, axis=0)
    print(
        f"\nCV mean — "
        f"Sens={mean_r[4]:.4f}, Spec={mean_r[5]:.4f}, "
        f"Acc={mean_r[6]:.4f}, MCC={mean_r[7]:.4f}, AUC={mean_r[8]:.4f}"
    )
    write_data.extend(mean_r)


# ── independent test mode ─────────────────────────────────────────────────────

if VALIDATION_MODE == "independent":
    x_train, y_train = handle_imbalance(IMBALANCE, x_train, y_train)
    gen = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)

    time_log("Start training")
    model = DeepScan(num_filters=NUM_FILTER, num_hidden=NUM_HIDDEN,
                     window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()
    model.fit(gen, epochs=EPOCHS, shuffle=True)
    time_log("End training")

    time_log("Start evaluation")
    res = model_test(model, x_test, y_test)
    write_data.extend(res)
    time_log("End evaluation")


# ── save results ──────────────────────────────────────────────────────────────

def save_csv(row, start_time):
    row.append(datetime.datetime.now() - start_time)
    out = "C:/jupyter/juan/AIP/code/results/PROTGPT2_PLM_KG_MCNN.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"Results saved: {out}")

save_csv(write_data, a)


# ── save model weights ────────────────────────────────────────────────────────

os.makedirs("./saved_weights/model", exist_ok=True)
weights_path = f"./saved_weights/model/AIP_esm2_mcnn_{MAXSEQ}_{WINDOW_SIZES}.h5"
model.save_weights(weights_path)
print(f"Weights saved: {weights_path}")
