# -*- coding: utf-8 -*-
"""
NN.py — Fraud Detection with Neural Network (Recall-First)
- Descomprime ZIP y lee CSV
- Preprocesa: OneHot para categóricas + StandardScaler en numéricas
- MLP (PyTorch) con pos_weight para clase minoritaria
- Early stopping por AP (PR-AUC) de validación
- Selección de umbral: piso de precisión o máximo F2
- Guarda artefactos: model.pt, preprocessor.joblib, threshold.json, metrics.csv
"""

import os
import json
import zipfile
import argparse
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ================== CONFIG EDITABLE ==================
TARGET_COL = "is_fraud"   # <-- CAMBIA al nombre de tu etiqueta (0/1)
ID_COLS = []              # p. ej., ["transaction_id"] si quieres excluirla
CAT_COLS = []             # si vacío, se infieren por dtype object/category
NUM_COLS = []             # si vacío, se infieren como el resto

TEST_SIZE = 0.2           # test estratificado
VAL_SIZE  = 0.2           # split interno del train para validación
RANDOM_STATE = 42

# Objetivo de umbral (elige solo uno)
PRECISION_FLOOR = 0.10    # 0.10, 0.20 ... (si no se cumple, cae a F2)
USE_F2_INSTEAD = False    # si True ignora PRECISION_FLOOR y usa F2

# Entrenamiento
MAX_EPOCHS = 50
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4       # L2
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 6    # epochs sin mejorar AP val

OUT_DIR = "./fraud_outputs_nn"
# ====================================================


def infer_feature_types(df: pd.DataFrame):
    global CAT_COLS, NUM_COLS
    if not CAT_COLS:
        CAT_COLS = [c for c in df.columns
                    if (df[c].dtype == 'object' or str(df[c].dtype).startswith('category'))
                    and c != TARGET_COL and c not in ID_COLS]
    if not NUM_COLS:
        NUM_COLS = [c for c in df.columns if c not in CAT_COLS + [TARGET_COL] + ID_COLS]


def make_preprocessor():
    transformers = []
    if NUM_COLS:
        transformers.append(("num", StandardScaler(), NUM_COLS))
    if CAT_COLS:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS))
    pre = ColumnTransformer(transformers, remainder="drop")
    return pre


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


class MLP(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        hidden1 = min(512, max(64, in_dim // 2))
        hidden2 = min(256, max(32, in_dim // 4))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)  # salida logit
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # [N]


@dataclass
class TrainResult:
    model_path: str
    preproc_path: str
    threshold_path: str
    metrics_csv: str


def pick_threshold_precision_floor(y_true, scores, precision_floor=0.1):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best_thr, best_rec, best_prec = 0.5, 0.0, 0.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= precision_floor and r > best_rec:
            best_rec, best_thr, best_prec = r, t, p
    return best_thr, best_rec, best_prec


def pick_threshold_fbeta(y_true, scores, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best_thr, best_fbeta, best_tuple = 0.5, -1.0, (0.0, 0.0)
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        denom = (beta**2) * p + r
        if denom > 0:
            fbeta = (1 + beta**2) * (p * r) / denom
            if fbeta > best_fbeta:
                best_fbeta, best_thr, best_tuple = fbeta, t, (p, r)
    return best_thr, best_fbeta, best_tuple


def train_nn(X_tr, y_tr, X_va, y_va, in_dim, device):
    torch.manual_seed(RANDOM_STATE)
    model = MLP(in_dim, dropout=DROPOUT).to(device)

    # pos_weight = (#neg / #pos) para BCEWithLogitsLoss
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    pos_weight_val = (neg / max(1, pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ds_tr = TensorDataset(to_tensor(X_tr, device), torch.tensor(y_tr, dtype=torch.float32, device=device))
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    best_ap, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        # Eval AP en validación
        model.eval()
        with torch.no_grad():
            va_logits = model(to_tensor(X_va, device))
            va_scores = torch.sigmoid(va_logits).detach().cpu().numpy()
        ap = average_precision_score(y_va, va_scores)

        improved = ap > best_ap + 1e-5
        print(f"Epoch {epoch:02d} | Val AP={ap:.4f} | "
              f"{'IMPROVED' if improved else f'no-improve ({no_improve+1})'}")

        if improved:
            best_ap = ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

    # cargar mejor estado
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    assert TARGET_COL in df.columns, f"No se encontró la columna objetivo '{TARGET_COL}'."
    df = df.drop(columns=ID_COLS, errors="ignore").copy()

    infer_feature_types(df)
    pre = make_preprocessor()

    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    # Split estratificado: train/test y dentro del train: train/val
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr, y_tr, test_size=VAL_SIZE, stratify=y_tr, random_state=RANDOM_STATE)

    # Fit preprocesador SOLO con train (X_tr2)
    X_tr2_mat = pre.fit_transform(X_tr2)
    X_va_mat  = pre.transform(X_va)
    X_te_mat  = pre.transform(X_te)

    in_dim = X_tr2_mat.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | Input dim: {in_dim}")

    # Entrena NN
    model = train_nn(X_tr2_mat, y_tr2, X_va_mat, y_va, in_dim, device)

    # Elegir umbral en validación
    model.eval()
    with torch.no_grad():
        va_scores = torch.sigmoid(model(to_tensor(X_va_mat, device))).cpu().numpy()

    if USE_F2_INSTEAD:
        thr, best_f2, (p, r) = pick_threshold_fbeta(y_va, va_scores, beta=2.0)
        thr_info = {"strategy": "F2", "threshold": float(thr), "F2": float(best_f2), "precision": float(p), "recall": float(r)}
        print(f"[Umbral F2] thr={thr:.6f} | F2={best_f2:.4f} | Prec={p:.4f} | Rec={r:.4f}")
    else:
        thr, r, p = pick_threshold_precision_floor(y_va, va_scores, precision_floor=PRECISION_FLOOR)
        if r == 0.0 and p == 0.0:
            thr, best_f2, (p, r) = pick_threshold_fbeta(y_va, va_scores, beta=2.0)
            thr_info = {"strategy": "F2_fallback", "threshold": float(thr), "F2": float(best_f2), "precision": float(p), "recall": float(r)}
            print(f"[Fallback F2] thr={thr:.6f} | F2={best_f2:.4f} | Prec={p:.4f} | Rec={r:.4f}")
        else:
            thr_info = {"strategy": f"precision_floor_{PRECISION_FLOOR}", "threshold": float(thr), "precision": float(p), "recall": float(r)}
            print(f"[Umbral Prec≥{PRECISION_FLOOR:.2f}] thr={thr:.6f} | Prec={p:.4f} | Rec={r:.4f}")

    # Evaluación final en TEST
    with torch.no_grad():
        te_scores = torch.sigmoid(model(to_tensor(X_te_mat, device))).cpu().numpy()
    y_pred = (te_scores >= thr_info["threshold"]).astype(int)

    ap_test = average_precision_score(y_te, te_scores)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    report = classification_report(y_te, y_pred, digits=4, output_dict=True)

    print("\n[TEST] PR-AUC: {:.4f}".format(ap_test))
    print("[TEST] Confusion Matrix (labels=[0,1]):\n", cm)
    print("[TEST] Classification Report @thr={:.6f}".format(thr_info["threshold"]))

    # Guardados
    model_path = os.path.join(OUT_DIR, "model.pt")
    preproc_path = os.path.join(OUT_DIR, "preprocessor.joblib")
    thr_path = os.path.join(OUT_DIR, "threshold.json")
    metrics_csv = os.path.join(OUT_DIR, "metrics_test.csv")

    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim}, model_path)
    joblib.dump(pre, preproc_path)
    with open(thr_path, "w") as f:
        json.dump(thr_info, f, indent=2)

    rep_df = pd.DataFrame(report).T
    rep_df["PR_AUC"] = ap_test
    rep_df.to_csv(metrics_csv, index=True)

    print(f"\nArtefactos guardados en: {OUT_DIR}")
    print(f"- Modelo: {model_path}")
    print(f"- Preprocesador: {preproc_path}")
    print(f"- Umbral: {thr_path}")
    print(f"- Métricas (test): {metrics_csv}")

    return TrainResult(model_path, preproc_path, thr_path, metrics_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, required=True, help="Ruta al ZIP con el dataset")
    parser.add_argument("--csv", type=str, required=True, help="Nombre del CSV dentro del ZIP")
    args = parser.parse_args()

    # Extraer ZIP a ./data_tmp
    os.makedirs("./data_tmp", exist_ok=True)
    with zipfile.ZipFile(args.zip, "r") as zf:
        zf.extractall("./data_tmp")

    df = pd.read_csv(os.path.join("./data_tmp", args.csv))
    main(df)
