import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import xgboost as xgb

# ====== RUTAS (COMBINED para train/valid; BASE para test) ======
CSV_COMB = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\train_balanced_combined_months_0_5.csv"
CSV_BASE = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\train_balanced_base_months_0_5.csv"
TARGET_COL = "fraud_bool"
RSEED = 42

def print_metrics(y_true, y_pred, y_proba, title=""):
    if title: print(f"\n===== {title} =====")
    for c in [0, 1]:
        p = precision_score(y_true, y_pred, pos_label=c)
        r = recall_score(y_true, y_pred, pos_label=c)
        f = f1_score(y_true, y_pred, pos_label=c)
        print(f"\nClase: {'No Fraude (0)' if c==0 else 'Fraude (1)'}")
        print("  Precision:", round(p, 4))
        print("  Recall:   ", round(r, 4))
        print("  F1 Score: ", round(f, 4))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nMatriz de confusión (crudos) [filas = verdadero, columnas = predicho]")
    print("              Pred 0   Pred 1")
    print(f"True 0 (NF)   {cm[0,0]:7d} {cm[0,1]:7d}")
    print(f"True 1 (F)    {cm[1,0]:7d} {cm[1,1]:7d}")

    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmn = np.nan_to_num(cmn)
    print("\nMatriz de confusión NORMALIZADA por fila")
    print("              Pred 0   Pred 1")
    print(f"True 0 (NF)   {cmn[0,0]:7.4f} {cmn[0,1]:7.4f}")
    print(f"True 1 (F)    {cmn[1,0]:7.4f} {cmn[1,1]:7.4f}")

    print(f"\nF1 macro:     {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 weighted:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"ROC-AUC:      {roc_auc_score(y_true, y_proba):.4f}")
    print(f"PR-AUC:       {average_precision_score(y_true, y_proba):.4f}")

# ====== Carga ======
comb = pd.read_csv(CSV_COMB)   # <-- COMBINED para train/valid
base = pd.read_csv(CSV_BASE)   # <-- BASE como test final

assert TARGET_COL in comb.columns and TARGET_COL in base.columns

# ====== Alineación de columnas ======
feat_comb = [c for c in comb.columns if c != TARGET_COL]
feat_base = [c for c in base.columns if c != TARGET_COL]
common_feats = sorted(set(feat_comb).intersection(feat_base))
if not common_feats:
    raise ValueError("No hay features en común entre COMBINED y BASE.")

Xc, yc = comb[common_feats], comb[TARGET_COL]
Xb_test, yb_test = base[common_feats], base[TARGET_COL]  # test = TODO el BASE

# ====== Split 80/20 en COMBINED (train/valid) ======
Xc_train, Xc_valid, yc_train, yc_valid = train_test_split(
    Xc, yc, test_size=0.20, random_state=RSEED, stratify=yc
)

# ====== Modelo (validación con el 20% de COMBINED) ======
model = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RSEED,
)

callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True, metric_name="logloss")]

model.fit(
    Xc_train, yc_train,
    eval_set=[(Xc_valid, yc_valid)],  # VALIDACIÓN = 20% de COMBINED
    #callbacks=callbacks,
    verbose=False
)

# ====== Reporte en VALID (20% COMBINED) ======
yc_pred = model.predict(Xc_valid)
yc_prob = model.predict_proba(Xc_valid)[:, 1]
print_metrics(yc_valid, yc_pred, yc_prob, title="VALIDACIÓN (20% COMBINED)")

# ====== Reporte en TEST (100% BASE) ======
yb_pred = model.predict(Xb_test)
yb_prob = model.predict_proba(Xb_test)[:, 1]
print_metrics(yb_test, yb_pred, yb_prob, title="TEST (BASE)")
