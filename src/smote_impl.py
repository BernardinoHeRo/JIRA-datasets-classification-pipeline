# src/smote_impl.py

import pandas as pd
import numpy as np
from typing import Tuple

from imblearn.over_sampling import SMOTE


def smote_balance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:

    # Asegurar tipos
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Distribución original
    class_counts = y_train.value_counts().sort_index()
    if verbose:
        print("Implementación de SMOTE Iniciada.")
        print("     Clase mayoritaria (0): ", class_counts.max())
        print("     Clase minoritaria (1): ", class_counts.min())
        print("     Total de clases: ", class_counts.sum())
        print("     Indice de desbalance: ", np.round(class_counts.max() / class_counts.min(), 2))
        
        # print("\n[SMOTE] Distribución original:")
        # print(f"        {class_counts.to_dict()}")

    if len(class_counts) != 2:
        raise ValueError("Este esquema SMOTE está pensado para clasificación binaria.")

    # Identificar clases
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    N = len(y_train)

    # Target ~ N/2 por clase (como en HCBOU)
    target_per_class = N // 2

    # Separar clases
    maj_mask = y_train == majority_class
    min_mask = y_train == minority_class

    X_maj = X_train.loc[maj_mask]
    y_maj = y_train.loc[maj_mask]
    X_min = X_train.loc[min_mask]
    y_min = y_train.loc[min_mask]

    # ----------------------------
    # 1) Submuestreo de mayoría
    # ----------------------------
    n_maj_target = min(target_per_class, len(X_maj))
    X_maj_down = X_maj.sample(n=n_maj_target, random_state=random_state)
    y_maj_down = y_maj.loc[X_maj_down.index]

    # if verbose:
    #    print(f"\n[SMOTE] Clase mayoritaria ({majority_class}): "
    #          f"{len(X_maj)} -> {len(X_maj_down)} muestras")

    # ----------------------------
    # 2) SMOTE sobre minoría
    # ----------------------------
    # Combinamos min + maj_down para que SMOTE vea ambas clases
    X_for_smote = pd.concat([X_min, X_maj_down], axis=0)
    y_for_smote = pd.concat([y_min, y_maj_down], axis=0)

    # Queremos que la minoría llegue a target_per_class
    sampling_strategy = {minority_class: target_per_class}

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    X_res, y_res = smote.fit_resample(X_for_smote, y_for_smote)

    # Después de SMOTE:
    final_counts = y_res.value_counts().sort_index()

    # Imprimir tamaño final del train balanceado
    print(f"\nTamaño final train balanceado: {X_res.shape[0]}")

    if verbose:
        print(f"Distribución clases: "f"{final_counts.to_dict()}")

    # Asegurar DataFrame/Serie
    X_bal = pd.DataFrame(X_res, columns=X_train.columns)
    y_bal = pd.Series(y_res, name=y_train.name if y_train.name else "target")

    return X_bal, y_bal