# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASETS_DIR,
    ARTIFACTS_DIR,
    TARGET_COL,
    COLUMNS_TO_DROP,
)


def preprocess_dataset(dataset_name: str):
    # ----------------------------
    # 1) Cargar el CSV
    # ----------------------------
    csv_path = DATASETS_DIR / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    print(f"[1] Preprocesamiento - Dataset original cargado: {dataset_name}")
    print(f"[1] Preprocesamiento - Dimensiones originales: {df.shape}")
    
    # ----------------------------
    # 2) Eliminar columnas no deseadas
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
    # Convertimos RealBug a int (0/1) por seguridad
    y = df[TARGET_COL].astype(int)

    # ----------------------------
    # 4) Hacer split Train/Test (80/20, estratificado)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"[1] Preprocesamiento - División estratificada 80/20 realizada.")
    print(f"[1] Preprocesamiento - X_train: {X_train.shape}, X_test: {X_test.shape}")

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
    print(f"[1] Preprocesamiento - Archivos CSV guardados en con éxito")

    return X_train, X_test, y_train, y_test