# src/utils/label_checks.py
from pathlib import Path
import pandas as pd
from src.config import PHASE_03_SCALING_DIR, BALANCING_METHODS, SCALING_TYPES

BUGGY_LABEL = 1

def assert_buggy_label_is_one(dataset_name: str):
    for method in BALANCING_METHODS:
        for scaler in SCALING_TYPES:
            base = PHASE_03_SCALING_DIR / method / dataset_name / scaler
            y_train_path = base / "y_train.csv"
            if not y_train_path.exists():
                continue

            y_train = pd.read_csv(y_train_path).squeeze()
            unique = set(y_train.unique())

            assert len(unique) == 2, (
                f"[{dataset_name} | {method} | {scaler}] "
                f"Esperaba problema binario, encontré {unique}"
            )
            assert unique <= {0, 1}, (
                f"[{dataset_name} | {method} | {scaler}] "
                f"Esperaba etiquetas en {{0,1}}, encontré {unique}"
            )
            assert (y_train == BUGGY_LABEL).sum() > 0, (
                f"[{dataset_name} | {method} | {scaler}] "
                f"No hay ejemplos de clase positiva {BUGGY_LABEL}"
            )

            print(
                f"✓ Etiquetas OK para {dataset_name} | {method} | {scaler} "
                f"(positivos={(y_train == 1).sum()}, negativos={(y_train == 0).sum()})"
            )