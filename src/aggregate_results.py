# src/aggregate_results_all.py

"""
Agrega TODOS los resultados finales de la Fase 5 en una sola tabla.

Para cada combinación:
    artifacts/05_final_models/<method>/<dataset>/<scaler_type>/<model>_final_metrics.json

Genera:
    1) results_all_metrics.csv        -> TODAS las métricas (para análisis interno)
    2) results_core_metrics.csv       -> Subconjunto limpio para tablas de resultados
    3) results_top3_per_dataset.csv   -> Top 3 configuraciones por dataset según un score compuesto
"""

import json
from pathlib import Path

import pandas as pd

from src.config import (
    PHASE_05_FINAL_DIR,
    DATASETS,
    BALANCING_METHODS,
    SCALING_TYPES,
)

# Etiquetas más amigables para tablas/figuras
MODEL_LABELS = {
    "svm": "SVM",
    "naive_bayes_gaussian": "Naive Bayes Gaussiano",
    "decision_tree": "Árbol de Decisión",
}


def load_all_final_metrics() -> pd.DataFrame:
    """
    Recorre todos los datasets, métodos de balanceo, tipos de escalado y modelos,
    y carga los archivos *_final_metrics.json en un único DataFrame.
    """
    rows = []

    models = ["svm", "naive_bayes_gaussian", "decision_tree"]

    for dataset in DATASETS:
        for method in BALANCING_METHODS:
            for scaler in SCALING_TYPES:
                base_dir = PHASE_05_FINAL_DIR / method / dataset / scaler

                for model_name in models:
                    metrics_path = base_dir / f"{model_name}_final_metrics.json"
                    if not metrics_path.exists():
                        # Puede que alguna combinación haya fallado y no tenga métricas finales
                        print(f"⚠ No existe: {metrics_path}")
                        continue

                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)

                    row = {
                        "dataset": dataset,
                        "balancing_method": method,
                        "scaler_type": scaler,
                        "model": model_name,
                        "model_label": MODEL_LABELS.get(model_name, model_name),
                    }

                    # Aplanar / adjuntar todas las métricas del JSON
                    for k, v in metrics.items():
                        # Si es dict/list (classification_report, best_hyperparameters, etc.),
                        # lo guardamos como JSON string para no perderlo.
                        if isinstance(v, (dict, list)):
                            row[k] = json.dumps(v)
                        else:
                            row[k] = v

                    rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = load_all_final_metrics()

    if df.empty:
        print("No se encontraron métricas finales para agregar.")
        return

    # ==================================================
    # 1) CSV COMPLETO (TODAS las métricas)
    # ==================================================
    preferred_first_cols = [
        "dataset",
        "balancing_method",
        "scaler_type",
        "model",
        "model_label",
        "F1_Score",
        "Precision",
        "Recall",
        "Exactitud",
        "Error",
        "ROC_AUC",
        "PR_AUC",
        "training_time_seconds",
    ]

    first_cols = [c for c in preferred_first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    df_full = df[first_cols + other_cols]

    out_full = PHASE_05_FINAL_DIR / "results_all_metrics.csv"
    out_full.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(out_full, index=False, encoding="utf-8")
    print(f"✅ Archivo maestro COMPLETO guardado en:\n   {out_full}")

    # ==================================================
    # 2) CSV “CORE” (métricas clave para tablas)
    # ==================================================
    core_cols_preferred = [
        "dataset",
        "balancing_method",
        "scaler_type",
        "model_label",

        # Conteos básicos (muy interpretables)
        "TP", "FP", "FN", "TN",

        # Métricas principales
        "F1_Score",
        "Precision",
        "Recall",          # = Sensibilidad = TP_Rate
        "Especificidad",
        "Error",
        "ROC_AUC",
        "PR_AUC",
    ]

    core_cols = [c for c in core_cols_preferred if c in df.columns]
    df_core = df[core_cols].copy()

    # Asegurarnos de que las métricas numéricas sean realmente numéricas
    metric_cols = [
        "TP", "FP", "FN", "TN",
        "F1_Score", "Precision", "Recall",
        "Especificidad", "Error", "ROC_AUC", "PR_AUC",
    ]
    metric_cols = [c for c in metric_cols if c in df_core.columns]
    df_core[metric_cols] = df_core[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    out_core = PHASE_05_FINAL_DIR / "results_core_metrics.csv"
    df_core.to_csv(out_core, index=False, encoding="utf-8")
    print(f"✅ Archivo RESUMIDO de métricas clave guardado en:\n   {out_core}")

    print("\nVista rápida (core, primeras 5 filas):")
    print(df_core.head())

    # ==================================================
    # 3) TOP 3 POR DATASET (según score compuesto)
    # ==================================================
    # 3.1 Filtro de configuraciones patológicas:
    #     - Especificidad muy baja (< 0.2)  -> casi todo se marca como defectuoso
    #     - Error muy alto (> 0.6)
    df_filtered = df_core.copy()

    if "Especificidad" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Especificidad"] >= 0.2]

    if "Error" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Error"] <= 0.6]

    if df_filtered.empty:
        print(
            "\n⚠ Después de filtrar por Especificidad >= 0.2 y Error <= 0.6 "
            "no quedaron filas. No se generará el TOP 3."
        )
        return

    # 3.2 Definir score compuesto:
    #     Puedes ajustar estos pesos según tu criterio:
    #     - 0.4 * Recall
    #     - 0.4 * F1_Score
    #     - 0.2 * PR_AUC
    for col in ["Recall", "F1_Score", "PR_AUC"]:
        if col not in df_filtered.columns:
            df_filtered[col] = float("nan")

    # Rellenar PR_AUC faltante (si hubiera) con 0.0 para no romper el cálculo
    df_filtered["PR_AUC"] = df_filtered["PR_AUC"].fillna(0.0)

    df_filtered["score"] = (
        0.4 * df_filtered["Recall"] +
        0.4 * df_filtered["F1_Score"] +
        0.2 * df_filtered["PR_AUC"]
    )

    # 3.3 Top 3 por dataset
    df_filtered = df_filtered.sort_values(
        ["dataset", "score"], ascending=[True, False]
    )

    top3 = (
        df_filtered
        .groupby("dataset", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )

    # Orden de columnas para el TOP 3 (centrado en interpretación)
    top3_cols_preferred = [
        "dataset",
        "balancing_method",
        "scaler_type",
        "model_label",
        "TP", "FP", "FN", "TN",
        "F1_Score",
        "Precision",
        "Recall",
        "Especificidad",
        "Error",
        "PR_AUC",
        "ROC_AUC",
        "score",
    ]
    top3_cols = [c for c in top3_cols_preferred if c in top3.columns]
    top3 = top3[top3_cols]

    out_top3 = PHASE_05_FINAL_DIR / "results_top3_per_dataset.csv"
    top3.to_csv(out_top3, index=False, encoding="utf-8")

    print(f"\n✅ Archivo TOP 3 por dataset guardado en:\n   {out_top3}")
    print("\nVista rápida (TOP 3, primeras filas):")
    print(top3.head())


if __name__ == "__main__":
    main()