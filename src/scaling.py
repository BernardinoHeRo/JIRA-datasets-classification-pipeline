# src/scaling.py

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.config import (
    PHASE_01_PREPROCESSING_DIR,      # artifacts/01_preprocessing
    PHASE_02_BALANCING_DIR,          # artifacts/02_balancing
    PHASE_03_SCALING_DIR,            # artifacts/03_scaling (SIN FS)
    PHASE_03_FEATURE_SELECTION_DIR,  # artifacts/03a_feature_selection
    PHASE_03_FS_SCALING_DIR          # artifacts/03b_scaling_fs (CON FS)
)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _get_scaler(scaler_type: str):
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Scaler no soportado: {scaler_type}")


def _align_test_to_train_columns(X_train: pd.DataFrame, X_test: pd.DataFrame, tag: str = ""):
    """
    Alinea X_test para que tenga EXACTAMENTE las mismas columnas que X_train:
    - Si X_test tiene columnas extra -> se eliminan
    - Si X_test le faltan columnas -> se crean con 0
    - Se reordena X_test con el mismo orden que X_train

    Esto evita errores tipo:
    ValueError: Feature names unseen at fit time: ...
    """
    extra_in_test = set(X_test.columns) - set(X_train.columns)
    missing_in_test = set(X_train.columns) - set(X_test.columns)

    if extra_in_test:
        print(f"    ⚠ [{tag}] Columnas extra en test (se eliminan):")
        print(f"       {sorted(extra_in_test)}")
        X_test = X_test.drop(columns=list(extra_in_test), errors="ignore")

    if missing_in_test:
        print(f"    ⚠ [{tag}] Columnas faltantes en test (se crean con 0):")
        print(f"       {sorted(missing_in_test)}")
        for c in missing_in_test:
            X_test[c] = 0

    # Reordenar para que coincida EXACTO con train
    X_test = X_test[X_train.columns]
    return X_test


# -------------------------------------------------------------------
# RUTA A: ESCALADO SIN SELECCIÓN DE CARACTERÍSTICAS
# -------------------------------------------------------------------

def scale_balanced(
    dataset_name: str,
    method: str,
    scaler_type: str = "standard"
):
    print(f"[3] [{method}] Escalando (SIN FS) {dataset_name} con {scaler_type}")

    # 1) Train balanceado (fase 2)
    bal_dir = PHASE_02_BALANCING_DIR / method / dataset_name
    X_train_bal = pd.read_csv(bal_dir / "X_train_bal.csv")
    y_train_bal = pd.read_csv(bal_dir / "y_train_bal.csv").squeeze()

    # 2) Test original (fase 1)
    prep_dir = PHASE_01_PREPROCESSING_DIR / dataset_name
    X_test = pd.read_csv(prep_dir / "X_test.csv")
    y_test = pd.read_csv(prep_dir / "y_test.csv").squeeze()

    # 3) Alinear columnas (EVITA el error de "Feature names unseen...")
    X_test = _align_test_to_train_columns(X_train_bal, X_test, tag="3-SIN-FS")

    # 4) Escalar
    scaler = _get_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # 5) Guardar
    out_dir = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X_train_bal.columns).to_csv(out_dir / "X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_train_bal.columns).to_csv(out_dir / "X_test_scaled.csv", index=False)

    y_train_bal.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)


# -------------------------------------------------------------------
# RUTA B: ESCALADO CON SELECCIÓN DE CARACTERÍSTICAS (FS)
# -------------------------------------------------------------------

def scale_fs_selected(
    dataset_name: str,
    method: str,
    scaler_type: str = "standard"
):
    print(f"[3-FS] [{method}] Escalando (con FS) {dataset_name} con {scaler_type}")

    # 1) Salida de FS
    fs_dir = PHASE_03_FEATURE_SELECTION_DIR / method / dataset_name
    X_train_fs = pd.read_csv(fs_dir / "X_train_fs.csv")
    X_test_fs = pd.read_csv(fs_dir / "X_test_fs.csv")
    y_train = pd.read_csv(fs_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(fs_dir / "y_test.csv").squeeze()

    # 2) Alinear columnas por seguridad (mismas columnas y orden)
    X_test_fs = _align_test_to_train_columns(X_train_fs, X_test_fs, tag="3-FS")

    # 3) Escalar
    scaler = _get_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train_fs)
    X_test_scaled = scaler.transform(X_test_fs)

    # 4) Guardar
    out_dir = PHASE_03_FS_SCALING_DIR / method / dataset_name / scaler_type
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=X_train_fs.columns).to_csv(out_dir / "X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_train_fs.columns).to_csv(out_dir / "X_test_scaled.csv", index=False)

    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)