# src/balancing.py

# ============================
# FASE 2: BALANCEO DE CLASES
# ============================
# Objetivo:
# - Cargar los datos preprocesados de la fase 1
#   (artifacts/01_preprocessing/<dataset>/X_train.csv, y_train.csv)
# - Aplicar distintos métodos de balanceo sobre el TRAIN:
#   * SIN BALANCEO (baseline)
#   * CSBBoost
#   * HCBOU
#   * SMOTE
# - Guardar los datasets balanceados en:
#   artifacts/02_balancing/<metodo>/<dataset>/
#       - X_train_bal.csv
#       - y_train_bal.csv
# ============================

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.csbboost_impl import csbboost_resample
from src.hcbou_impl import hcbou_balance, get_recommended_params
from src.smote_impl import smote_balance


def _load_preprocessed(dataset_name: str):
    """
    Carga X_train, X_test, y_train, y_test desde la fase 1
    (01_preprocessing).

    Esta función es interna (por eso el guion bajo).
    """

    base = ARTIFACTS_DIR / "01_preprocessing" / dataset_name

    # Leemos los CSV que generó preprocessing.py
    X_train = pd.read_csv(base / "X_train.csv")
    X_test = pd.read_csv(base / "X_test.csv")
    # squeeze('columns') convierte DataFrame de una sola columna en Series
    y_train = pd.read_csv(base / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(base / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test


# ----------------------------
# "BALANCEO" SIN RESAMPLING (BASELINE)
# ----------------------------

def balance_without_resampling(dataset_name: str):
    """
    Genera la versión SIN balanceo (baseline).

    - Lee X_train, y_train de 01_preprocessing
    - Se queda solo con columnas numéricas (igual que CSBBoost/HCBOU/SMOTE)
    - No aplica ningún método de resampling
    - Guarda en artifacts/02_balancing/unbalanced/<dataset>/:
        - X_train_bal.csv
        - y_train_bal.csv
    """

    print(f"\n\n========== 'Balanceo' [UNBALANCED] con {dataset_name} ==========")
    print("================================================================")

    # 1) Cargar datos preprocesados
    X_train, X_test, y_train, y_test = _load_preprocessed(dataset_name)
    print(f"[2] 'Balanceo' [UNBALANCED] - Carga de datos preprocesados completa.")

    # 2) Utilizar únicamente columnas numéricas
    #    Mantiene consistencia con el resto de métodos de balanceo.
    X_train_num = X_train.select_dtypes(include=[np.number])
    print(f"[2] 'Balanceo' [UNBALANCED] - Selección de columnas numéricas.")
    # print(f"[2] 'Balanceo' [UNBALANCED] - Columnas numéricas usadas: {list(X_train_num.columns)}")

    # 3) No se aplica ningún resampling:
    #    Simplemente usamos el train original como "train balanceado"
    X_bal = X_train_num
    y_bal = y_train

    # 4) Carpeta de salida para el baseline sin balanceo
    out_dir = ARTIFACTS_DIR / "02_balancing" / "unbalanced" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Guardar train "balanceado" (realmente sin tocar)
    X_bal.to_csv(out_dir / "X_train_bal.csv", index=False)
    y_bal.to_csv(out_dir / "y_train_bal.csv", index=False)

    print(f"[2] 'Balanceo' [UNBALANCED] - Archivos CSV guardados con éxito.")
    return X_bal, y_bal


# ----------------------------
# BALANCEO CON CSBBOOST
# ----------------------------

def balance_with_csbboost(dataset_name: str):
    """
    Aplica CSBBoost al conjunto de entrenamiento de un dataset.

    - Lee X_train, y_train de 01_preprocessing
    - Se queda solo con columnas numéricas
    - Aplica csbboost_resample
    - Guarda en artifacts/02_balancing/csbboost/<dataset>/
    """

    print(f"\n\n========== Ballanceo [CSBBoost] con {dataset_name} ==========")
    print("=============================================================")

    # 1) Cargar datos preprocesados
    X_train, X_test, y_train, y_test = _load_preprocessed(dataset_name)
    print(f"[2] Balanceo [CSBBoost] - Carga de datos preprocesados completa.")

    # 2) Utilizar únicamente columnas numéricas
    X_train_num = X_train.select_dtypes(include=[np.number])
    print(f"[2] Balanceo [CSBBoost] - Selección de columnas numéricas.")

    # 3) Aplicar CSBBoost (función ya implementada en csbboost_impl.py)
    X_bal, y_bal = csbboost_resample(
        X_train_num,
        y_train,
        k_neighbors=5,
        k_range_major=(2, 10),
        k_range_minor=(2, 10),
        random_state=42,
    )
    print(f"\n[2] Balanceo [CSBBoost] - CSBBoost completado.")

    # 4) Carpeta de salida para CSBBoost
    out_dir = ARTIFACTS_DIR / "02_balancing" / "csbboost" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Guardar train balanceado
    X_bal.to_csv(out_dir / "X_train_bal.csv", index=False)
    y_bal.to_csv(out_dir / "y_train_bal.csv", index=False)
    print(f"[2] Balanceo [CSBBoost] - Archivos CSV guardados con éxito.")

    return X_bal, y_bal


# ----------------------------
# BALANCEO CON HCBOU
# ----------------------------

def balance_with_hcbou(dataset_name: str):
    """
    Aplica HCBOU al conjunto de entrenamiento de un dataset.

    - Lee X_train, y_train de 01_preprocessing
    - Se queda solo con columnas numéricas
    - Obtiene parámetros recomendados con get_recommended_params
    - Aplica hcbou_balance
    - Guarda en artifacts/02_balancing/hcbou/<dataset>/
    """

    print(f"\n\n========== Balanceo [HCBOU] con {dataset_name} ==========")
    print("=========================================================")

    # 1) Cargar datos preprocesados
    X_train, X_test, y_train, y_test = _load_preprocessed(dataset_name)
    print(f"[2] Balanceo [HCBOU] - Carga de datos preprocesados completa.")

    # 2) Sólo columnas numéricas (como recomienda el método)
    X_train_num = X_train.select_dtypes(include=[np.number])
    print(f"[2] Balanceo [HCBOU] - Selección de columnas numéricas.")

    # 3) Obtener parámetros recomendados para HCBOU
    params = get_recommended_params(
        X_train_num,
        y_train,
        scenario="binary_classification",
    )
    print(f"[2] Balanceo [HCBOU] - Parámetros: {params}")

    # 4) Aplicar HCBOU
    X_bal, y_bal = hcbou_balance(
        X_train_num,
        y_train,
        **params,
        random_state=42,
        verbose=True,
    )
    print(f"\n[2] Balanceo [HCBOU] - HCBOU completado.")

    # 5) Carpeta de salida para HCBOU
    out_dir = ARTIFACTS_DIR / "02_balancing" / "hcbou" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Guardar train balanceado
    X_bal.to_csv(out_dir / "X_train_bal.csv", index=False)
    y_bal.to_csv(out_dir / "y_train_bal.csv", index=False)
    print(f"[2] Balanceo [HCBOU] - Archivos CSV guardados con éxito.")
    return X_bal, y_bal


# ----------------------------
# BALANCEO CON SMOTE
# ----------------------------

def balance_with_smote(dataset_name: str):
    """
    Aplica SMOTE al conjunto de entrenamiento de un dataset.

    - Lee X_train, y_train de 01_preprocessing
    - Se queda solo con columnas numéricas (como en CSBBoost y HCBOU)
    - Aplica smote_balance
    - Guarda en artifacts/02_balancing/smote/<dataset>/
    """

    print(f"\n\n========== Balanceo [SMOTE] con {dataset_name} ==========")
    print("=========================================================")

    # 1) Cargar datos preprocesados (fase 1)
    pre_dir = ARTIFACTS_DIR / "01_preprocessing" / dataset_name
    X_train = pd.read_csv(pre_dir / "X_train.csv")
    y_train = pd.read_csv(pre_dir / "y_train.csv").squeeze("columns")

    print(f"[2] Balanceo [SMOTE] - Carga de datos preprocesados completa.")

    # 2) Solo columnas numéricas (igual que en CSBBoost/HCBOU)
    X_train_num = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    )
    print(f"[2] Balanceo [SMOTE] - Selección de columnas numéricas.")

    # 3) Aplicar SMOTE con el criterio N/2 por clase
    X_train_bal, y_train_bal = smote_balance(
        X_train_num,
        y_train,
        random_state=42,
        verbose=True,
    )

    # 4) Carpeta de salida para SMOTE
    out_dir = ARTIFACTS_DIR / "02_balancing" / "smote" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Guardar resultados balanceados
    X_train_bal.to_csv(out_dir / "X_train_bal.csv", index=False)
    y_train_bal.to_csv(out_dir / "y_train_bal.csv", index=False)

    print(f"\n[2] Balanceo [SMOTE] - SMOTE completado.")
    print(f"[2] Balanceo [SMOTE] - Archivos CSV guardados con éxito.")
    return X_train_bal, y_train_bal