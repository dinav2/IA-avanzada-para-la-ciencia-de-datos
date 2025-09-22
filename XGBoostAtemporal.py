import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
import xgboost as xgb

# ======== RUTAS (ACTUALIZADAS) ========
# Cambia SOLO la carpeta base si es necesario
TRAIN_CSV = r"C:\Users\rpzda\Documents\python varios\ML_GDL\train_random_03.csv"        # ENTRENAMIENTO
VALID_CSV = r"C:\Users\rpzda\Documents\python varios\ML_GDL\validation_random_03.csv"  # VALIDACIÓN
TEST_CSV  = r"C:\Users\rpzda\Documents\python varios\ML_GDL\test_random_03.csv"        # TEST (con etiqueta)

TARGET_COL = "fraud_bool"
RSEED = 42

# ======== UMBRAL MANUAL ========
CHOSEN_THR = 0.20   # <<<<< CAMBIA AQUÍ EL UMBRAL

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
df_train = pd.read_csv(TRAIN_CSV)
df_valid = pd.read_csv(VALID_CSV)
df_test  = pd.read_csv(TEST_CSV)

# Validaciones
assert TARGET_COL in df_train.columns, f"{TARGET_COL} no está en TRAIN_CSV"
assert TARGET_COL in df_valid.columns, f"{TARGET_COL} no está en VALID_CSV"
assert TARGET_COL in df_test.columns,  f"{TARGET_COL} no está en TEST_CSV"

# ======== 2) ALINEAR FEATURES ========
feat_train = [c for c in df_train.columns if c != TARGET_COL]
feat_valid = [c for c in df_valid.columns if c != TARGET_COL]
feat_test  = [c for c in df_test.columns  if c != TARGET_COL]

common_feats = sorted(set(feat_train).intersection(set(feat_valid)).intersection(set(feat_test)))
if not common_feats:
    raise ValueError("No hay features en común entre TRAIN, VALID y TEST.")

X_train = df_train[common_feats]
y_train = df_train[TARGET_COL]
X_val   = df_valid[common_feats]
y_val   = df_valid[TARGET_COL]
X_test  = df_test[common_feats]
y_test  = df_test[TARGET_COL]

# ======== 3) MODELO ========
model = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=RSEED,
)

callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True, metric_name="logloss")]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    #callbacks=callbacks,   # <- déjalo comentado si no quieres ES
    verbose=False
)

# ======== 4) VALIDACIÓN ========
val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= CHOSEN_THR).astype(int)   # <<<<< UMBRAL APLICADO
print_metrics(y_val, val_pred, val_proba, title=f"VALIDACIÓN (validation_random_03.csv, umbral={CHOSEN_THR})")

# ======== 5) TEST ========
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= CHOSEN_THR).astype(int)  # <<<<< UMBRAL APLICADO
print_metrics(y_test, test_pred, test_proba, title=f"TEST (test_random_03.csv, umbral={CHOSEN_THR})")
