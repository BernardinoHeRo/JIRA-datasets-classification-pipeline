# src/model_selection.py
# =============================================================================
# FASE 6: SELECCIÓN DEL MEJOR MODELO
# =============================================================================
# Objetivo:
#   - Leer los resultados finales de la Fase 5 (final_training.py).
#   - Reconstruir / asegurar métricas clave (incluyendo MCC y BER).
#   - Construir un ranking de modelos por:
#       * dataset
#       * método de balanceo
#       * tipo de escalado
#       * modelo (SVM, Naive Bayes, Decision Tree)
#   - Seleccionar el MEJOR modelo para cada dataset bajo el criterio:
#       1) Maximizar Recall (clase defectuosa / Buggy).
#       2) Maximizar MCC.
#       3) Minimizar BER.
#       4) Maximizar Precision.
#       5) Maximizar Exactitud (Accuracy).
#   - Guardar:
#       * CSV con ranking completo por dataset.
#       * JSON con el mejor modelo y sus métricas.
#   - 🆕 Construir una tabla GLOBAL de “lo mejor de lo mejor”:
#       * Un renglón por dataset con su mejor modelo.
#       * Guardar en: artifacts/06_model_selection/_GLOBAL_best_models.csv
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PHASE_05_FINAL_DIR, BALANCING_METHODS, SCALING_TYPES

# =============================================================================
# CONSTANTES Y ESTRUCTURAS AUXILIARES
# =============================================================================

# Directorio base donde se guardará la selección de modelos
PHASE_06_MODEL_SELECTION_DIR: Path = PHASE_05_FINAL_DIR.parent / "06_model_selection"
PHASE_06_MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)

# Modelos considerados
MODEL_NAMES: List[str] = ["svm", "naive_bayes_gaussian", "decision_tree"]


@dataclass(frozen=True)
class ModelConfigKey:
    """Identificador único de una configuración de modelo final."""

    dataset: str
    method: str
    scaler_type: str
    model_name: str

    @property
    def as_tuple(self) -> Tuple[str, str, str, str]:
        return (self.dataset, self.method, self.scaler_type, self.model_name)


# =============================================================================
# 1. UTILIDADES PARA RECONSTRUIR MÉTRICAS (MCC, BER, ETC.)
# =============================================================================


def _compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calcula el MCC (Matthews Correlation Coefficient) a partir de TP, TN, FP, FN.

    Fórmula:
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Si el denominador es 0 (p.ej. una clase no aparece), se devuelve 0.0.
    """
    numerator = tp * tn - fp * fn
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if denom <= 0:
        return 0.0

    return float(numerator / np.sqrt(denom))


def _compute_ber(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calcula el Balanced Error Rate (BER).

    Definición habitual:
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        BER = 1 - 0.5 * (TPR + TNR)

    Si alguna clase no está presente (división por 0), TPR/TNR se tratan como 0.
    """
    p = tp + fn
    n = tn + fp

    tpr = tp / p if p > 0 else 0.0  # Sensibilidad
    tnr = tn / n if n > 0 else 0.0  # Especificidad

    ber = 1.0 - 0.5 * (tpr + tnr)
    return float(ber)


def _ensure_confusion_components(metrics: Dict) -> Tuple[int, int, int, int]:
    """
    Asegura que existan TP, TN, FP, FN en el diccionario de métricas.
    Si no existen, intenta reconstruirlos desde 'confusion_matrix'.

    La matriz de confusión se asume:
        [[TN, FP],
         [FN, TP]]
    """
    tp = metrics.get("TP")
    tn = metrics.get("TN")
    fp = metrics.get("FP")
    fn = metrics.get("FN")

    if None not in (tp, tn, fp, fn):
        return int(tp), int(tn), int(fp), int(fn)

    # Si faltan, intentamos reconstruir desde 'confusion_matrix'
    cm = metrics.get("confusion_matrix")
    if cm is None:
        raise ValueError("No se encontraron TP/TN/FP/FN ni 'confusion_matrix' en métricas.")

    cm_arr = np.array(cm)
    if cm_arr.shape != (2, 2):
        raise ValueError(
            f"Se esperaba confusion_matrix 2x2, pero se obtuvo shape={cm_arr.shape}"
        )

    tn, fp, fn, tp = cm_arr.ravel()
    return int(tp), int(tn), int(fp), int(fn)


def _ensure_additional_metrics(metrics: Dict) -> Dict:
    """
    Asegura que las métricas contengan:
        - TP, TN, FP, FN
        - MCC
        - BER

    Si MCC/BER no están, se calculan y se agregan al diccionario.
    """
    tp, tn, fp, fn = _ensure_confusion_components(metrics)

    metrics["TP"] = tp
    metrics["TN"] = tn
    metrics["FP"] = fp
    metrics["FN"] = fn

    if "MCC" not in metrics:
        metrics["MCC"] = _compute_mcc(tp, tn, fp, fn)

    if "BER" not in metrics:
        metrics["BER"] = _compute_ber(tp, tn, fp, fn)

    return metrics


# =============================================================================
# 2. CARGA DE MÉTRICAS FINALES (FASE 5)
# =============================================================================


def _load_final_metrics_for_config(
    dataset_name: str,
    method: str,
    scaler_type: str,
) -> Dict[ModelConfigKey, Dict]:
    """
    Carga los archivos *_final_metrics.json de la Fase 5 para una configuración
    fija de (dataset, método de balanceo, tipo de escalado).

    Retorna:
        diccionario:
            key: ModelConfigKey
            value: dict de métricas (con MCC y BER asegurados)
    """
    base_dir = PHASE_05_FINAL_DIR / method / dataset_name / scaler_type
    results: Dict[ModelConfigKey, Dict] = {}

    if not base_dir.exists():
        # Nada entrenado para esta combinación
        return results

    for model_name in MODEL_NAMES:
        metrics_file = base_dir / f"{model_name}_final_metrics.json"
        if not metrics_file.exists():
            # El modelo no se entrenó o falló; se ignora
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"    ⚠ Error leyendo métricas de {metrics_file}: {e}")
            continue

        # Asegurar TP, TN, FP, FN, MCC, BER
        try:
            metrics = _ensure_additional_metrics(metrics)
        except Exception as e:
            print(f"    ⚠ Error asegurando métricas adicionales para {metrics_file}: {e}")
            continue

        key = ModelConfigKey(
            dataset=dataset_name,
            method=method,
            scaler_type=scaler_type,
            model_name=model_name,
        )
        results[key] = metrics

    return results


def _load_all_final_metrics_for_dataset(
    dataset_name: str,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str, str], Dict]]:
    """
    Carga todas las métricas finales para un dataset dado, abarcando:
        - todos los métodos de balanceo (BALANCING_METHODS)
        - todos los tipos de escalado (SCALING_TYPES)
        - todos los modelos definidos en MODEL_NAMES

    Retorna:
        ranking_df: DataFrame con una fila por modelo-configuración.
        metrics_by_key: dict para recuperar métricas completas por clave.
    """
    rows: List[Dict] = []
    metrics_by_key: Dict[Tuple[str, str, str, str], Dict] = {}

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            config_metrics = _load_final_metrics_for_config(dataset_name, method, scaler_type)
            if not config_metrics:
                continue

            for key, metrics in config_metrics.items():
                # Guardamos métrica completa para uso posterior
                metrics_by_key[key.as_tuple] = metrics

                # Creamos una fila "plana" para el ranking (sin objetos pesados)
                row = {
                    "dataset": key.dataset,
                    "balancing_method": key.method,
                    "scaler_type": key.scaler_type,
                    "model_name": key.model_name,
                    # Métricas clave
                    "Recall": metrics.get("Recall", 0.0),
                    "F2_Score": metrics.get("F2_Score", None),
                    "Precision": metrics.get("Precision", 0.0),
                    "Exactitud": metrics.get("Exactitud", 0.0),
                    "F1_Score": metrics.get("F1_Score", 0.0),
                    "Sensibilidad": metrics.get(
                        "Sensibilidad", metrics.get("Recall", 0.0)
                    ),
                    "Especificidad": metrics.get("Especificidad", 0.0),
                    "Error": metrics.get("Error", 0.0),
                    "TP_Rate": metrics.get("TP_Rate", metrics.get("Recall", 0.0)),
                    "FP_Rate": metrics.get("FP_Rate", 0.0),
                    "MCC": metrics.get("MCC", 0.0),
                    "BER": metrics.get("BER", 0.5),
                    "ROC_AUC": metrics.get("ROC_AUC", None),
                    "TP": metrics.get("TP", None),
                    "TN": metrics.get("TN", None),
                    "FP": metrics.get("FP", None),
                    "FN": metrics.get("FN", None),
                }

                rows.append(row)

    if not rows:
        # No hay resultados para este dataset
        return pd.DataFrame(), metrics_by_key

    ranking_df = pd.DataFrame(rows)

    return ranking_df, metrics_by_key


# =============================================================================
# 3. CRITERIO DE SELECCIÓN DEL MEJOR MODELO
# =============================================================================


def _selection_score(row: pd.Series) -> Tuple[float, float, float, float, float]:
    """
    Define el criterio de ordenamiento para seleccionar el mejor modelo.

    Orden de importancia (de mayor a menor):
        1) Recall (clase defectuosa / Buggy)  -> máx
        2) MCC                                -> máx
        3) BER                                -> mín  (por eso usamos -BER)
        4) Precision                          -> máx
        5) Exactitud                          -> máx
    """
    recall = float(row.get("Recall", 0.0))
    mcc = float(row.get("MCC", 0.0))
    ber = float(row.get("BER", 0.5))
    precision = float(row.get("Precision", 0.0))
    accuracy = float(row.get("Exactitud", 0.0))

    # Nota: como queremos minimizar BER, usamos -BER en la tupla
    return (recall, mcc, -ber, precision, accuracy)


def _select_best_row(ranking_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    A partir de un DataFrame con todas las configuraciones, selecciona la
    mejor fila según el criterio definido en _selection_score.

    Si ranking_df está vacío, devuelve None.
    """
    if ranking_df.empty:
        return None

    # Aplicamos el score a cada fila
    scores = ranking_df.apply(_selection_score, axis=1)
    # Obtenemos el índice de la fila con mejor score (orden lexicográfico)
    best_idx = scores.idxmax()
    best_row = ranking_df.loc[best_idx]

    return best_row


# =============================================================================
# 4. GUARDADO DE RESULTADOS (RANKING COMPLETO + MEJOR MODELO)
# =============================================================================


def _save_full_ranking(dataset_name: str, ranking_df: pd.DataFrame) -> Path:
    """
    Guarda el ranking completo de modelos para un dataset en CSV.

    Ruta:
        artifacts/06_model_selection/<dataset>_full_ranking.csv
    """
    PHASE_06_MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PHASE_06_MODEL_SELECTION_DIR / f"{dataset_name}_full_ranking.csv"
    ranking_df.to_csv(csv_path, index=False)
    return csv_path


def _save_best_model_summary(
    dataset_name: str,
    best_row: pd.Series,
    metrics_full: Dict,
) -> Path:
    """
    Guarda un resumen del mejor modelo para el dataset en formato JSON.

    Ruta:
        artifacts/06_model_selection/<dataset>_best_model.json

    Contenido:
        - dataset
        - selección (modelo, balanceo, escalado)
        - métricas completas
        - explicación del criterio de selección
    """
    PHASE_06_MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PHASE_06_MODEL_SELECTION_DIR / f"{dataset_name}_best_model.json"

    summary = {
        "dataset": dataset_name,
        "selected_model": {
            "model_name": best_row["model_name"],
            "balancing_method": best_row["balancing_method"],
            "scaler_type": best_row["scaler_type"],
        },
        "selection_criteria": {
            "description": (
                "Modelo seleccionado maximizando Recall (clase defectuosa / Buggy), "
                "luego MCC, luego minimizando BER, y en caso de empate usando "
                "Precision y Exactitud como criterios secundarios."
            ),
            "priority_order": [
                "Recall (Buggy) – prioridad máxima (no dejar pasar defectuosos reales)",
                "MCC – calidad global del clasificador considerando ambas clases",
                "BER – menor es mejor; balance de errores en No-Buggy y Buggy",
                "Precision (Buggy) – evitar excesivos falsos positivos",
                "Exactitud (Accuracy) – criterio de desempate adicional",
            ],
        },
        "metrics": metrics_full,
        "timestamp": datetime.now().isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return json_path


# =============================================================================
# 5. FUNCIÓN PRINCIPAL PÚBLICA: select_best_models(dataset_name)
# =============================================================================


def select_best_models(dataset_name: str):
    """
    Ejecuta la Fase 6 para un dataset:

        1) Carga todas las métricas finales de Fase 5.
        2) Reconstruye MCC y BER si no estaban calculados.
        3) Construye un ranking completo de:
                dataset, método_balanceo, escalado, modelo.
        4) Selecciona el mejor modelo con el criterio Recall→MCC→BER→Precision→Exactitud.
        5) Guarda:
                - <dataset>_full_ranking.csv
                - <dataset>_best_model.json
        6) Imprime en consola un pequeño resumen.

    Uso típico desde run_all.py:
        from src.model_selection import select_best_models

        ...
        # Después de train_final_models(dataset)
        select_best_models(dataset)
    """
    print(f"\n{'=' * 80}")
    print(f"FASE 6: SELECCIÓN DEL MEJOR MODELO - DATASET: {dataset_name}")
    print(f"{'=' * 80}")

    # 1) Cargar todos los resultados de Fase 5
    ranking_df, metrics_by_key = _load_all_final_metrics_for_dataset(dataset_name)

    if ranking_df.empty:
        print(f"  ✗ No se encontraron métricas finales para el dataset: {dataset_name}")
        return None

    # 2) Guardar ranking completo (para análisis posterior)
    ranking_path = _save_full_ranking(dataset_name, ranking_df)
    print(f"  ✓ Ranking completo guardado en: {ranking_path}")

    # 3) Seleccionar la mejor configuración
    best_row = _select_best_row(ranking_df)
    if best_row is None:
        print(f"  ✗ No se pudo seleccionar un modelo para el dataset: {dataset_name}")
        return None

    key = (
        best_row["dataset"],
        best_row["balancing_method"],
        best_row["scaler_type"],
        best_row["model_name"],
    )
    best_metrics_full = metrics_by_key.get(key, {})

    # 4) Guardar resumen del mejor modelo
    best_json_path = _save_best_model_summary(dataset_name, best_row, best_metrics_full)
    print(f"  ✓ Mejor modelo guardado en: {best_json_path}")

    # 5) Imprimir resumen compacto en consola
    print("\n  ► MEJOR MODELO SELECCIONADO")
    print(f"     - Dataset           : {dataset_name}")
    print(f"     - Modelo            : {best_row['model_name']}")
    print(f"     - Balanceo          : {best_row['balancing_method']}")
    print(f"     - Escalado          : {best_row['scaler_type']}")
    print("     - Métricas clave:")
    print(f"         Recall (Buggy)  : {best_row['Recall']:.4f}")
    f2 = best_row.get("F2_Score", None)
    if f2 is not None:
        print(f"         F2_Score        : {float(f2):.4f}")
    print(f"         Precision       : {best_row['Precision']:.4f}")
    print(f"         F1_Score        : {best_row['F1_Score']:.4f}")
    print(f"         MCC             : {best_row['MCC']:.4f}")
    print(f"         BER (↓ mejor)   : {best_row['BER']:.4f}")
    print(f"         Exactitud       : {best_row['Exactitud']:.4f}")
    print(f"         Error           : {best_row['Error']:.4f}")

    print(f"{'=' * 80}\n")

    return {
        "ranking": ranking_df,
        "best_row": best_row,
        "best_metrics": best_metrics_full,
    }


# =============================================================================
# 6. NUEVO PASO: TABLA GLOBAL “LO MEJOR DE LO MEJOR”
# =============================================================================


def build_global_best_table(dataset_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Construye una tabla global con el MEJOR modelo de cada dataset.

    Lee:
        artifacts/06_model_selection/<dataset>_best_model.json

    Genera:
        artifacts/06_model_selection/_GLOBAL_best_models.csv

    Columnas típicas:
        - dataset
        - model_name
        - balancing_method
        - scaler_type
        - Recall
        - F2_Score (si existe)
        - Precision
        - F1_Score
        - MCC
        - BER
        - Exactitud
        - Error
    """
    rows: List[Dict] = []

    for ds in dataset_names:
        json_path = PHASE_06_MODEL_SELECTION_DIR / f"{ds}_best_model.json"
        if not json_path.exists():
            # Puede ser que ese dataset no haya llegado a Fase 6 correctamente
            print(f"  ⚠ No se encontró best_model.json para dataset: {ds}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Error leyendo {json_path}: {e}")
            continue

        selected = data.get("selected_model", {})
        metrics = data.get("metrics", {})

        row = {
            "dataset": ds,
            "model_name": selected.get("model_name"),
            "balancing_method": selected.get("balancing_method"),
            "scaler_type": selected.get("scaler_type"),
            "Recall": metrics.get("Recall"),
            "F2_Score": metrics.get("F2_Score"),
            "Precision": metrics.get("Precision"),
            "F1_Score": metrics.get("F1_Score"),
            "MCC": metrics.get("MCC"),
            "BER": metrics.get("BER"),
            "Exactitud": metrics.get("Exactitud"),
            "Error": metrics.get("Error"),
            "ROC_AUC": metrics.get("ROC_AUC"),
        }

        rows.append(row)

    if not rows:
        print("  ✗ No se pudieron construir filas para la tabla global de mejores modelos.")
        return None

    df = pd.DataFrame(rows)

    # Guardar CSV global
    PHASE_06_MODEL_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    global_csv_path = PHASE_06_MODEL_SELECTION_DIR / "_GLOBAL_best_models.csv"
    df.to_csv(global_csv_path, index=False)

    print(f"\n  ✓ Tabla GLOBAL de 'lo mejor de lo mejor' guardada en:")
    print(f"      {global_csv_path}\n")

    return df


# =============================================================================
# 7. MODO SCRIPT (para pruebas independientes)
# =============================================================================

if __name__ == "__main__":
    import sys
    from src.config import DATASETS

    args = sys.argv[1:]

    # Para saber qué datasets sí se procesaron y luego construir la tabla global
    processed_datasets: List[str] = []

    if args:
        if args == ["all"]:
            for ds in DATASETS:
                res = select_best_models(ds)
                if res is not None:
                    processed_datasets.append(ds)
        else:
            for ds in args:
                if ds not in DATASETS:
                    print(f"⚠ Dataset '{ds}' no está en DATASETS de config.py. Se ignora.")
                else:
                    res = select_best_models(ds)
                    if res is not None:
                        processed_datasets.append(ds)
    else:
        # Sin argumentos: aplicar a todos los datasets de config.py
        if DATASETS:
            for ds in DATASETS:
                res = select_best_models(ds)
                if res is not None:
                    processed_datasets.append(ds)
        else:
            print("No hay datasets definidos en config.py")

    # 🆕 Paso global: construir tabla de “lo mejor de lo mejor”
    if processed_datasets:
        build_global_best_table(processed_datasets)