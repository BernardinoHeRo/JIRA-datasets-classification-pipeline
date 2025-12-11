# src/preprocessing.py

# ============================
# FASE 1: PREPROCESAMIENTO
# ============================
# Objetivo:
# - Cargar el CSV original de cada dataset
# - Eliminar columnas que no se usarán como predictores
# - Separar X (predictores) e y (target RealBug)
# - Partir en train/test 80/20 estratificado
# - Guardar los 4 archivos en:
#   artifacts/01_preprocessing/<dataset>/
#       - X_train.csv
#       - X_test.csv
#       - y_train.csv
#       - y_test.csv
# ============================

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASETS_DIR,
    ARTIFACTS_DIR,
    TARGET_COL,
    COLUMNS_TO_DROP,
)


def preprocess_dataset(dataset_name: str):
    """
    Ejecuta el preprocesamiento básico para un dataset.

    Parámetros
    ----------
    dataset_name : str
        Nombre del dataset SIN la extensión .csv
        (por ejemplo 'activemq-5.0.0').

    Retorna
    -------
    X_train, X_test, y_train, y_test : DataFrames/Series
        Conjuntos de entrenamiento y prueba, ya separados.
    """

    # ----------------------------
    # 1) Cargar el CSV original
    # ----------------------------
    csv_path = DATASETS_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    print(f"[1] Preprocesamiento - Dataset original cargado: {dataset_name}")
    print(f"[1] Preprocesamiento - Dimensiones originales: {df.shape}")

    # ----------------------------
    # 2) Eliminar columnas no deseadas
    #    (solo si existen en el DataFrame)
    # ----------------------------
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[1] Preprocesamiento - Columnas eliminadas: {cols_to_drop}")
        print(f"[1] Preprocesamiento - Dimensiones nuevas: {df.shape}")

    # ----------------------------
    # 3) Separar X (features) e y (target)
    # ----------------------------
    X = df.drop(columns=[TARGET_COL])

    # Aseguramos que el target esté en formato 0/1
    y_raw = df[TARGET_COL]

    # Si ya es numérico 0/1, esto basta:
    # (si en algún dataset viniera como bool o similar, se puede extender el mapping)
    y = y_raw.astype(int)

    # Verificación extra: asegurar que las etiquetas sean {0,1}
    unique_labels = sorted(y.unique().tolist())
    print(f"[1] Preprocesamiento - Etiquetas únicas en y: {unique_labels}")
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"[1] ERROR: La columna '{TARGET_COL}' contiene valores fuera de {{0,1}}: "
            f"{unique_labels}. Normaliza el target antes de continuar."
        )

    # ----------------------------
    # 4) Hacer split Train/Test (80/20, estratificado)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # mantiene proporción de clases
    )
    print(f"[1] Preprocesamiento - División estratificada 80/20 realizada.")
    print(f"[1] Preprocesamiento - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(
        f"[1] Preprocesamiento - Distribución y_train: "
        f"{y_train.value_counts(normalize=True).to_dict()}"
    )
    print(
        f"[1] Preprocesamiento - Distribución y_test:  "
        f"{y_test.value_counts(normalize=True).to_dict()}"
    )

    # ----------------------------
    # 5) Crear carpeta de salida de la fase 1
    #    artifacts/01_preprocessing/<dataset>/
    # ----------------------------
    out_dir = ARTIFACTS_DIR / "01_preprocessing" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 6) Guardar los archivos en CSV
    # ----------------------------
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)
    print(f"[1] Preprocesamiento - Archivos CSV guardados con éxito en {out_dir}")

    # Devolvemos también en memoria por si quieres usarlo inmediatamente
    return X_train, X_test, y_train, y_test