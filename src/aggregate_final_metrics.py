# src/aggregate_final_metrics.py

import os
import json
from pathlib import Path

import pandas as pd


# Carpeta base donde están todos los modelos finales
BASE_DIR = Path("artifacts/05_final_models")


def iter_final_metrics():
    """
    Recorre recursivamente artifacts/05_final_models y genera una lista de filas
    (dicts) con:
        - route: fs / nofs
        - balancing: csbboost / hcbou / smote / unbalanced
        - dataset: nombre del dataset (activemq-5.0.0, etc.)
        - scaler: standard / robust
        - model: decision_tree / naive_bayes_gaussian / svm
        - métricas numéricas leídas desde final_metrics.json
    """
    rows = []

    for root, dirs, files in os.walk(BASE_DIR):
        if "final_metrics.json" not in files:
            continue

        metrics_path = Path(root) / "final_metrics.json"

        # Esperamos estructura:
        # artifacts/05_final_models/{route}/{balancing}/{dataset}/{scaler}/{model}/final_metrics.json
        rel_parts = metrics_path.relative_to(BASE_DIR).parts

        # rel_parts -> [route, balancing, dataset, scaler, model, "final_metrics.json"]
        if len(rel_parts) < 6:
            print(f"⚠️  Saltando ruta inesperada: {metrics_path}")
            continue

        route = rel_parts[0]          # fs / nofs
        balancing = rel_parts[1]      # csbboost / hcbou / smote / unbalanced
        dataset = rel_parts[2]
        scaler = rel_parts[3]         # standard / robust
        model = rel_parts[4]          # decision_tree / naive_bayes_gaussian / svm

        # Cargar métricas
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  No se pudo leer JSON: {metrics_path}")
            continue

        row = {
            "route": route,
            "balancing": balancing,
            "dataset": dataset,
            "scaler": scaler,
            "model": model,
        }

        # Nos quedamos con las métricas numéricas (accuracy, recall, mcc, ber, etc.)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v

        rows.append(row)

    return rows


def aggregate_final_metrics(
    output_csv: str = "artifacts/all_results/final_models_metrics.csv",
) -> None:
    """
    Construye un DataFrame con todas las combinaciones y lo guarda en un CSV global.
    """
    rows = iter_final_metrics()

    if not rows:
        print("⚠️  No se encontraron archivos final_metrics.json en "
              "artifacts/05_final_models/")
        return

    df = pd.DataFrame(rows)

    # Orden sugerido para que sea fácil de leer/filtrar
    sort_cols = [c for c in ["dataset", "balancing", "scaler", "route", "model"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("✅ Métricas agregadas correctamente.")
    print(f"   Archivo: {output_path}")
    print("\nPrimeras filas:\n")
    print(df.head())


if __name__ == "__main__":
    aggregate_final_metrics()