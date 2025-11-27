# src/scaling.py

# ============================
# FASE 3: ESCALADO (NORMALIZACIÓN)
# ============================
# Objetivo:
# - Usar los datos balanceados (fase 2):
#       X_train_bal, y_train_bal
# - Usar el test original (fase 1):
#       X_test, y_test
# - Aplicar StandardScaler o RobustScaler SOLO con info del TRAIN
# - Guardar salidas en:
#   artifacts/03_scaling/<method>/<dataset>/<scaler_type>/
# ============================

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.config import ARTIFACTS_DIR


def scale_balanced(dataset_name: str, method: str, scaler_type: str = "standard"):
    """
    Escala los datos usando StandardScaler o RobustScaler.

    Parámetros
    ----------
    dataset_name : str
        Nombre del dataset (sin .csv)
    method : str
        Método de balanceo ("csbboost", "hcbou")
    scaler_type : str
        Tipo de escalador:
        - "standard"
        - "robust"

    Flujo:
    1. Lee X_train_bal, y_train_bal (fase 2)
    2. Lee X_test, y_test (fase 1)
    3. Ajusta escalador SOLO con TRAIN balanceado
    4. Transforma TRAIN y TEST
    5. Guarda resultados organizados
    """

    print(f"[3] [{method}] Escalando {dataset_name} con {scaler_type}")

    # ----------------------------
    # 1) Cargar TRAIN balanceado (fase 2)
    # ----------------------------
    bal_dir = ARTIFACTS_DIR / "02_balancing" / method / dataset_name
    X_train_bal = pd.read_csv(bal_dir / "X_train_bal.csv")
    y_train_bal = pd.read_csv(bal_dir / "y_train_bal.csv").squeeze("columns")

    # ----------------------------
    # 2) Cargar TEST original (fase 1)
    # ----------------------------
    pre_dir = ARTIFACTS_DIR / "01_preprocessing" / dataset_name
    X_test = pd.read_csv(pre_dir / "X_test.csv")
    y_test = pd.read_csv(pre_dir / "y_test.csv").squeeze("columns")

    # El test debe tener EXACTAMENTE las columnas del train balanceado
    X_test_num = X_test[X_train_bal.columns]

    # ----------------------------
    # 3) Seleccionar el escalador
    # ----------------------------
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Tipo de escalador desconocido: {scaler_type}")

    # ----------------------------
    # 4) Ajustar el escalador usando SOLO TRAIN balanceado
    # ----------------------------
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test_num)

    # ----------------------------
    # 5) Crear carpeta de salida limpia
    # artifacts/03_scaling/<method>/<dataset>/<scaler_type>/
    # ----------------------------
    out_dir = ARTIFACTS_DIR / "03_scaling" / method / dataset_name / scaler_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 6) Guardar resultados
    # ----------------------------
    pd.DataFrame(X_train_scaled, columns=X_train_bal.columns).to_csv(
        out_dir / "X_train_scaled.csv",
        index=False,
    )
    pd.DataFrame(X_test_scaled, columns=X_train_bal.columns).to_csv(
        out_dir / "X_test_scaled.csv",
        index=False,
    )
    y_train_bal.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    # print(f"       → Escalado guardado en: {out_dir}")

    return out_dir