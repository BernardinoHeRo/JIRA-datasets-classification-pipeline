# src/final_training.py

from pathlib import Path
import json
import pickle
from datetime import datetime
from tabulate import tabulate
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    PHASE_03_SCALING_DIR,
    PHASE_03_FS_SCALING_DIR,
    PHASE_04_HYPERPARAMETER_DIR,
    PHASE_04_HYPERPARAMETER_FS_DIR,
    PHASE_05_FINAL_DIR,
    BALANCING_METHODS,
    SCALING_TYPES,
)


# =====================================================================
# UTILIDADES DE CARGA
# =====================================================================

def _load_scaled_data_for_final(dataset_name: str, method: str, scaler_type: str, use_fs: bool):
    
    base_root = PHASE_03_FS_SCALING_DIR if use_fs else PHASE_03_SCALING_DIR
    base_path = base_root / method / dataset_name / scaler_type

    # ruta_desc = "CON FS" if use_fs else "SIN FS"
    # print(f"   [F5] Cargando datos escalados ({ruta_desc}) desde: {base_path}")

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze("columns")

    print(f"[F5] Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def _get_hyperparam_root(use_fs: bool) -> Path:
    
    return PHASE_04_HYPERPARAMETER_FS_DIR if use_fs else PHASE_04_HYPERPARAMETER_DIR


def _get_final_root(use_fs: bool) -> Path:
    
    sub = "fs" if use_fs else "nofs"
    return PHASE_05_FINAL_DIR / sub


# =====================================================================
# CÁLCULO DE MÉTRICAS A PARTIR DE LA MATRIZ DE CONFUSIÓN
# =====================================================================

def _metrics_from_confusion(y_test, y_pred):
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        raise ValueError(f"Matriz de confusión inesperada: {cm}")

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    P = tp + fn
    N = tn + fp
    P_pred = tp + fp

    # Evitar divisiones por cero
    def _safe(div_num, div_den):
        return float(div_num / div_den) if div_den > 0 else 0.0

    error = _safe(fp + fn, total)
    exactitud = _safe(tn + tp, total)
    tp_rate = _safe(tp, P)
    fp_rate = _safe(fp, N)
    precision = _safe(tp, P_pred)
    recall = tp_rate
    sensibilidad = tp_rate
    especificidad = _safe(tn, N)

    metrics = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "P": int(P),
        "N": int(N),
        "Total": int(total),

        "Error": error,
        "Exactitud": exactitud,
        "TP_Rate": tp_rate,
        "FP_Rate": fp_rate,
        "Precision": precision,
        "Recall": recall,
        "Sensibilidad": sensibilidad,
        "Especificidad": especificidad,

        "F1_Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "Accuracy_sklearn": float(accuracy_score(y_test, y_pred)),
    }

    # Consistencias básicas
    # (No uso asserts fuertes para no romper ejecución por flotantes)
    return metrics, cm

def _print_confusion_and_metrics(model_name: str, metrics: dict, cm):
    """Imprime matriz de confusión y métricas en formato tabular (horizontal)."""
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*70}")
    print(f"MATRIZ DE CONFUSIÓN Y MÉTRICAS - {model_name.upper()}")
    print(f"{'='*70}")

    # ==================================================
    # 1) MATRIZ DE CONFUSIÓN
    # ==================================================
    cm_table = [
        ["Real \\ Pred", "No-Buggy", "Buggy"],
        ["No-Buggy", tn, fp],
        ["Buggy", fn, tp],
    ]

    print("\nMatriz de Confusión:")
    print(
        tabulate(
            cm_table,
            headers="firstrow",
            tablefmt="grid",
            stralign="center",
            numalign="center",
        )
    )

    # ==================================================
    # 2) MÉTRICAS PRINCIPALES (HORIZONTAL)
    # ==================================================
    metrics_headers = [
        "TP", "TN", "FP", "FN",
        "Total", "P (Buggy)", "N (No-Buggy)",
        "Error", "Exactitud",
        "Precision", "Recall",
        "F1-Score", "Especificidad",
        "TP Rate", "FP Rate"
    ]

    metrics_values = [
        tp, tn, fp, fn,
        metrics["Total"], metrics["P"], metrics["N"],
        f"{metrics['Error']:.4f}", f"{metrics['Exactitud']:.4f}",
        f"{metrics['Precision']:.4f}", f"{metrics['Recall']:.4f}",
        f"{metrics['F1_Score']:.4f}", f"{metrics['Especificidad']:.4f}",
        f"{metrics['TP_Rate']:.4f}", f"{metrics['FP_Rate']:.4f}",
    ]

    print("\nMétricas:")
    print(
        tabulate(
            [metrics_values],
            headers=metrics_headers,
            tablefmt="grid",
            stralign="center",
            numalign="center",
        )
    )

    print(f"{'='*70}\n")


# =====================================================================
# ENTRENAMIENTO / EVALUACIÓN FINAL PARA UNA CONFIGURACIÓN
# =====================================================================

def train_final_models_for_config(
    dataset_name: str,
    method: str,
    scaler_type: str,
    use_fs: bool = False,
):
    
    ruta_tag = "CON FS" if use_fs else "SIN FS"
    print(f"\n{'-'*80}")
    print(f"[F5] CONFIGURACIÓN FINAL ({ruta_tag}): {dataset_name} | {method} | {scaler_type}")
    print(f"{'-'*80}")

    # 1) Cargar datos escalados (train/test)
    try:
        X_train, X_test, y_train, y_test = _load_scaled_data_for_final(
            dataset_name, method, scaler_type, use_fs=use_fs
        )
    except FileNotFoundError as e:
        print(f"   ✗ [F5] No se encontraron datos escalados para esta configuración: {e}")
        return

    # 2) Raíz de hiperparámetros (FASE 4) y destino final (FASE 5)
    hyper_root = _get_hyperparam_root(use_fs)
    final_root = _get_final_root(use_fs)

    hp_dir = hyper_root / method / dataset_name / scaler_type
    out_dir = final_root / method / dataset_name / scaler_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # print(f"   [F5] Leyendo mejores modelos desde: {hp_dir}")
    # print(f"   [F5] Guardando resultados finales en: {out_dir}")

    model_names = ["svm", "naive_bayes_gaussian", "decision_tree"]
    summary = {}

    for model_name in model_names:
        model_file = hp_dir / f"{model_name}_best_model.pkl"
        params_file = hp_dir / f"{model_name}_best_params.json"

        if not model_file.exists() or not params_file.exists():
            print(f"   ⚠️ [F5] {model_name}: no se encontraron archivos de mejor modelo/params. Se omite.")
            summary[model_name] = None
            continue

        # 3) Cargar modelo entrenado (best_estimator_ de GridSearch)
        with open(model_file, "rb") as f:
            best_model = pickle.load(f)

        # 4) Evaluar en test
        print(f"[F5] Evaluando modelo final: {model_name}")
        y_pred = best_model.predict(X_test)
        metrics, cm = _metrics_from_confusion(y_test, y_pred)

        # 5) Cargar info de hiperparámetros (para incluir en el JSON final)
        with open(params_file, "r") as f:
            params_info = json.load(f)

        # Añadir info extra
        metrics["best_hyperparameters"] = params_info.get("best_params", {})
        metrics["cv_best_score"] = params_info.get("best_cv_score", None)
        metrics["timestamp"] = datetime.now().isoformat()
        metrics["confusion_matrix"] = cm.tolist()

        # 6) Guardar métricas por modelo
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        with open(model_out / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 7) Guardar matriz de confusión también en CSV cómodo para tesis
        cm_df = pd.DataFrame(
            cm,
            index=["real_0", "real_1"],
            columns=["pred_0", "pred_1"],
        )
        cm_df.to_csv(model_out / "confusion_matrix.csv")

        # 8) Guardar classification_report en txt (útil para inspección)
        report_txt = classification_report(y_test, y_pred, zero_division=0)
        with open(model_out / "classification_report.txt", "w") as f:
            f.write(report_txt)

        # 9) Imprimir bonito en consola
        _print_confusion_and_metrics(model_name, metrics, cm)

        # Para el resumen global
        summary[model_name] = metrics

    # 10) Guardar resumen por configuración
    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[F5] Configuración final completada: {dataset_name} | {method} | {scaler_type} ({ruta_tag})")


# =====================================================================
# WRAPPER POR DATASET (TODOS LOS MÉTODOS Y SCALERS)
# =====================================================================

def train_final_models(dataset_name: str, use_fs: bool = False):
    
    ruta_tag = "CON FS" if use_fs else "SIN FS"

    print(f"\n{'='*80}")
    print(f"FASE 5: ENTRENAMIENTO FINAL Y EVALUACIÓN [{ruta_tag}]")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            train_final_models_for_config(
                dataset_name=dataset_name,
                method=method,
                scaler_type=scaler_type,
                use_fs=use_fs,
            )

    print(f"\n{'='*80}")
    print(f"TODAS LAS CONFIGURACIONES FINALES COMPLETADAS PARA: {dataset_name} [{ruta_tag}]")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    from src.config import DATASETS

    if len(DATASETS) > 0:
        # Prueba rápida con el primer dataset SIN FS
        train_final_models(DATASETS[0], use_fs=False)
        # Para probar solo la ruta B:
        # train_final_models(DATASETS[0], use_fs=True)