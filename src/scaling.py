# src/scaling.py

# ============================
# FASE 3: ESCALADO (NORMALIZACIÓN)
# ============================
# Objetivo:
# - Usar los datos balanceados (fase 2):
#       X_train_bal, y_train_bal
# - Usar el test original (fase 1):
#       X_test, y_test
# - Aplicar StandardScaler o RobustScaler SOLO con info del TRAIN balanceado
# - Guardar salidas en:
#   artifacts/03_scaling/<method>/<dataset>/<scaler_type>/
# ============================

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.config import ARTIFACTS_DIR


def scale_balanced(dataset_name: str, method: str, scaler_type: str = "standard") -> Path:
    """
    Escala los datos usando StandardScaler o RobustScaler.

    Parámetros
    ----------
    dataset_name : str
        Nombre del dataset (sin .csv), por ejemplo:
        "activemq-5.0.0", "hbase-0.94.0", etc.
    method : str
        Método de balanceo (fase 2):
        - "unbalanced"  (baseline sin resampling)
        - "csbboost"
        - "hcbou"
        - "smote"
    scaler_type : str
        Tipo de escalador:
        - "standard" -> sklearn.preprocessing.StandardScaler
        - "robust"   -> sklearn.preprocessing.RobustScaler

    Flujo
    -----
    1. Lee X_train_bal, y_train_bal (fase 2).
    2. Lee X_test, y_test (fase 1).
    3. Ajusta el escalador SOLO con X_train_bal.
    4. Transforma X_train_bal y X_test.
    5. Guarda resultados en:
       artifacts/03_scaling/<method>/<dataset>/<scaler_type>/.

    Retorna
    -------
    out_dir : pathlib.Path
        Ruta a la carpeta donde se guardaron los archivos escalados.
    """

    print(f"[3] [{method}] Escalando {dataset_name} con {scaler_type}")

    # ----------------------------
    # 1) Cargar TRAIN balanceado (fase 2)
    # ----------------------------
    bal_dir = ARTIFACTS_DIR / "02_balancing" / method / dataset_name

    X_train_bal_path = bal_dir / "X_train_bal.csv"
    y_train_bal_path = bal_dir / "y_train_bal.csv"

    if not X_train_bal_path.exists() or not y_train_bal_path.exists():
        raise FileNotFoundError(
            f"No se encontraron los archivos balanceados para '{dataset_name}' "
            f"y método '{method}' en {bal_dir}"
        )

    X_train_bal = pd.read_csv(X_train_bal_path)
    y_train_bal = pd.read_csv(y_train_bal_path).squeeze("columns")

    # ----------------------------
    # 2) Cargar TEST original (fase 1)
    # ----------------------------
    pre_dir = ARTIFACTS_DIR / "01_preprocessing" / dataset_name

    X_test_path = pre_dir / "X_test.csv"
    y_test_path = pre_dir / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"No se encontraron X_test/y_test para '{dataset_name}' en {pre_dir}"
        )

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # El test debe tener EXACTAMENTE las columnas del train balanceado
    missing_cols = set(X_train_bal.columns) - set(X_test.columns)
    if missing_cols:
        raise ValueError(
            f"X_test de '{dataset_name}' no contiene todas las columnas de X_train_bal.\n"
            f"Faltan columnas: {sorted(missing_cols)}"
        )

    X_test_num = X_test[X_train_bal.columns]

    # ----------------------------
    # 3) Seleccionar el escalador
    # ----------------------------
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Tipo de escalador desconocido: {scaler_type!r}")

    # ----------------------------
    # 4) Ajustar el escalador usando SOLO TRAIN balanceado
    # ----------------------------
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test_num)

    # ----------------------------
    # 5) Crear carpeta de salida
    #    artifacts/03_scaling/<method>/<dataset>/<scaler_type>/
    # ----------------------------
    out_dir = ARTIFACTS_DIR / "03_scaling" / method / dataset_name / scaler_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 6) Guardar resultados
    # ----------------------------
    pd.DataFrame(
        X_train_scaled,
        columns=X_train_bal.columns,
    ).to_csv(out_dir / "X_train_scaled.csv", index=False)

    pd.DataFrame(
        X_test_scaled,
        columns=X_train_bal.columns,
    ).to_csv(out_dir / "X_test_scaled.csv", index=False)

    y_train_bal.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    return out_dir