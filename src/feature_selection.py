# src/feature_selection.py

from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    PHASE_01_PREPROCESSING_DIR,        # artifacts/01_preprocessing
    PHASE_02_BALANCING_DIR,            # artifacts/02_balancing
    PHASE_03_FEATURE_SELECTION_DIR     # artifacts/03a_feature_selection
)


def _get_fs_output_dir(method: str, dataset_name: str) -> Path:
    out_dir = PHASE_03_FEATURE_SELECTION_DIR / method / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_balanced_train(method: str, dataset_name: str):
    bal_dir = PHASE_02_BALANCING_DIR / method / dataset_name
    X_train_bal = pd.read_csv(bal_dir / "X_train_bal.csv")
    y_train_bal = pd.read_csv(bal_dir / "y_train_bal.csv").squeeze()
    return X_train_bal, y_train_bal


def _load_original_test(dataset_name: str):
    prep_dir = PHASE_01_PREPROCESSING_DIR / dataset_name
    X_test = pd.read_csv(prep_dir / "X_test.csv")
    y_test = pd.read_csv(prep_dir / "y_test.csv").squeeze()
    return X_test, y_test


def _align_test_to_train_columns(X_train: pd.DataFrame, X_test: pd.DataFrame, tag: str = "FS"):
    """
    Alinea test a columnas de train (igual que en scaling):
    - drop columnas extra en test
    - crear con 0 columnas faltantes en test
    - reordenar columnas como train
    """
    extra_in_test = set(X_test.columns) - set(X_train.columns)
    missing_in_test = set(X_train.columns) - set(X_test.columns)

    if extra_in_test:
        print(f"      ⚠ [{tag}] Columnas extra en test (se eliminan):")
        print(f"        {sorted(extra_in_test)}")
        X_test = X_test.drop(columns=list(extra_in_test), errors="ignore")

    if missing_in_test:
        print(f"      ⚠ [{tag}] Columnas faltantes en test (se crean con 0):")
        print(f"        {sorted(missing_in_test)}")
        for c in missing_in_test:
            X_test[c] = 0

    X_test = X_test[X_train.columns]
    return X_test


def _select_top_k_features(
    X_train: pd.DataFrame,
    y_train,
    k: int = 20,
    random_state: int = 42
):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_names = list(X_train.columns)

    indices = importances.argsort()[::-1]
    top_indices = indices[:min(k, len(feature_names))]

    selected_features = [feature_names[i] for i in top_indices]
    return selected_features


def select_features_for_balanced(
    dataset_name: str,
    method: str,
    k: int = 20
):
    print(f"   🔎 [FS] Dataset: {dataset_name} | Método: {method}")

    # 1) Train balanceado
    X_train_bal, y_train_bal = _load_balanced_train(method, dataset_name)
    print(f"      → {X_train_bal.shape[1]} features originales en X_train_bal.")

    # 2) Test original
    X_test, y_test = _load_original_test(dataset_name)

    # 3) Alinear test a train (evita columnas como 'File' u otras)
    X_test = _align_test_to_train_columns(X_train_bal, X_test, tag="FS")

    # 4) Seleccionar top-k (según importancia en RF)
    selected_features = _select_top_k_features(X_train_bal, y_train_bal, k=k)
    print(f"      → Seleccionadas top {min(k, len(selected_features))} features.")
    print(f"      → Características seleccionadas: {selected_features}")

    # 5) Aplicar selección (aquí ya no puede faltar nada)
    X_train_fs = X_train_bal[selected_features]
    X_test_fs = X_test[selected_features]

    # 6) Guardar
    out_dir = _get_fs_output_dir(method, dataset_name)

    X_train_fs.to_csv(out_dir / "X_train_fs.csv", index=False)
    X_test_fs.to_csv(out_dir / "X_test_fs.csv", index=False)
    y_train_bal.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    with open(out_dir / "selected_features.json", "w") as f:
        json.dump({"selected_features": selected_features}, f, indent=2)