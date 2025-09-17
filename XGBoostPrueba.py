
"""
# train_combined_valid_base_test_combined.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import xgboost as xgb

# ======== RUTAS ========
TRAIN_COMBINED = "C:\\Users\\rpzda\\Documents\\python varios\\ML_GDL\\IA-avanzada-para-la-ciencia-de-datos\\train_balanced_base_months_0_5\\train_balanced_combined_months_0_5.csv" # ENTRENAMIENTO
VALID_BASE     = "C:\\Users\\rpzda\\Documents\\python varios\\ML_GDL\\IA-avanzada-para-la-ciencia-de-datos\\train_balanced_base_months_0_5\\train_balanced_base_months_0_5.csv" # VALIDACIÓN (con label)
XTEST_COMB     = "C:\\Users\\rpzda\\Documents\\python varios\\ML_GDL\\IA-avanzada-para-la-ciencia-de-datos\\train_balanced_base_months_0_5\\X_test_base_months_0_5.csv"            # TEST (features)
YTEST_COMB     = "C:\\Users\\rpzda\\Documents\\python varios\\ML_GDL\\IA-avanzada-para-la-ciencia-de-datos\\train_balanced_base_months_0_5\\Y_test_base_months_0_5.csv"            # TEST (labels)

# Si tu columna objetivo no se llama "label", cámbiala aquí:
TARGET_COL = "fraud_bool"
RSEED = 42

def print_metrics(y_true, y_pred, y_proba, title=""):
    if title:
        print(f"\n===== {title} =====")
    for c in [0, 1]:
        p = precision_score(y_true, y_pred, pos_label=c)
        r = recall_score(y_true, y_pred, pos_label=c)
        f = f1_score(y_true, y_pred, pos_label=c)
        nombre = "No Fraude (0)" if c == 0 else "Fraude (1)"
        print(f"\nClase: {nombre}")
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

# ======== 1) CARGA ========
df_train = pd.read_csv(TRAIN_COMBINED)
df_valid = pd.read_csv(VALID_BASE)
X_test   = pd.read_csv(XTEST_COMB)
y_test   = pd.read_csv(YTEST_COMB).squeeze()  # convierte a Serie

# Validaciones rápidas
assert TARGET_COL in df_train.columns, f"{TARGET_COL} no está en TRAIN_COMBINED"
assert TARGET_COL in df_valid.columns, f"{TARGET_COL} no está en VALID_BASE"
# y_test puede venir con nombre de columna; si es DataFrame con 1 col, squeeze() lo vuelve Serie.

# ======== 2) ALINEAR FEATURES ========
feat_train = [c for c in df_train.columns if c != TARGET_COL]
feat_valid = [c for c in df_valid.columns if c != TARGET_COL]
common_feats = sorted(list(set(feat_train).intersection(set(feat_valid)).intersection(set(X_test.columns))))

if not common_feats:
    raise ValueError("No hay features en común entre TRAIN_COMBINED, VALID_BASE y X_test_combined.")

# Subconjuntos alineados y mismo orden de columnas
X_train = df_train[common_feats]
y_train = df_train[TARGET_COL]

X_val   = df_valid[common_feats]
y_val   = df_valid[TARGET_COL]

# Asegurar orden de columnas en X_test
X_test  = X_test[common_feats]

# ======== 3) MODELO + EARLY STOPPING (VALIDACIÓN EN BASE) ========
model = XGBClassifier(
    n_estimators=6500, #4000 | #6000
    learning_rate=0.15, #0.09 | #0.12
    max_depth=7, #7 | #9
    subsample=0.9, #0.9 |
    colsample_bytree=0.9, #0.9 |
    eval_metric="logloss",
    random_state=RSEED,
)

callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True, metric_name="logloss")]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # validación con BASE
    #callbacks=callbacks,         # si tu versión no soporta callbacks, comenta esta línea
    verbose=False
)

# ======== 4) REPORTE VALIDACIÓN (BASE) ========
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]
print_metrics(y_val, y_val_pred, y_val_prob, title="VALIDACIÓN (BASE)")

# ======== 5) REPORTE TEST (COMBINED) ========
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]
print_metrics(y_test, y_test_pred, y_test_prob, title="TEST (COMBINED)")
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import xgboost as xgb

# ======== RUTAS ========
TRAIN_COMBINED = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\train_balanced_combined_months_0_5.csv"  # ENTRENAMIENTO
VALID_BASE     = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\train_balanced_base_months_0_5.csv"      # VALIDACIÓN
XTEST_COMB     = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\X_test_base_months_0_5.csv"               # TEST FEATURES
YTEST_COMB     = r"C:\Users\rpzda\Documents\python varios\ML_GDL\IA-avanzada-para-la-ciencia-de-datos\train_balanced_base_months_0_5\Y_test_base_months_0_5.csv"               # TEST LABELS

TARGET_COL = "fraud_bool"
RSEED = 42

# ======== UMBRAL MANUAL ========
CHOSEN_THR = 0.1   # <<<<< CAMBIA AQUÍ EL UMBRAL

def print_metrics(y_true, y_pred, y_proba, title=""):
    if title:
        print(f"\n===== {title} =====")
    for c in [0, 1]:
        p = precision_score(y_true, y_pred, pos_label=c, zero_division=0)
        r = recall_score(y_true, y_pred, pos_label=c, zero_division=0)
        f = f1_score(y_true, y_pred, pos_label=c, zero_division=0)
        nombre = "No Fraude (0)" if c == 0 else "Fraude (1)"
        print(f"\nClase: {nombre}")
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

# ======== 1) CARGA ========
df_train = pd.read_csv(TRAIN_COMBINED)
df_valid = pd.read_csv(VALID_BASE)
X_test   = pd.read_csv(XTEST_COMB)
y_test   = pd.read_csv(YTEST_COMB).squeeze()

# Validaciones
assert TARGET_COL in df_train.columns, f"{TARGET_COL} no está en TRAIN_COMBINED"
assert TARGET_COL in df_valid.columns, f"{TARGET_COL} no está en VALID_BASE"

# ======== 2) ALINEAR FEATURES ========
feat_train = [c for c in df_train.columns if c != TARGET_COL]
feat_valid = [c for c in df_valid.columns if c != TARGET_COL]
common_feats = sorted(set(feat_train).intersection(set(feat_valid)).intersection(set(X_test.columns)))
if not common_feats:
    raise ValueError("No hay features en común entre TRAIN_COMBINED, VALID_BASE y X_test.")

X_train = df_train[common_feats]
y_train = df_train[TARGET_COL]
X_val   = df_valid[common_feats]
y_val   = df_valid[TARGET_COL]
X_test  = X_test[common_feats]

# ======== 3) MODELO ========
model = XGBClassifier(
    n_estimators=7000,
    learning_rate=0.12,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RSEED,
)

callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True, metric_name="logloss")]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    #callbacks=callbacks,
    verbose=False
)

# ======== 4) VALIDACIÓN ========
val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= CHOSEN_THR).astype(int)   # <<<<< UMBRAL APLICADO
print_metrics(y_val, val_pred, val_proba, title=f"VALIDACIÓN (BASE, umbral={CHOSEN_THR})")

# ======== 5) TEST ========
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= CHOSEN_THR).astype(int)  # <<<<< UMBRAL APLICADO
print_metrics(y_test, test_pred, test_proba, title=f"TEST (BASE, umbral={CHOSEN_THR})")
