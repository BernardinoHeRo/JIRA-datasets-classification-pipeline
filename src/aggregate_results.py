# src/aggregate_results.py

"""
Agregador de resultados de clasificadores (RUTA A y RUTA B) en un solo CSV.

Recorre recursivamente artifacts/ buscando CSV que tengan al menos estas columnas:

    - dataset
    - balancing_method
    - scaler_type
    - model_label

Opcionalmente pueden tener:
    - pipeline  (A / B, baseline / fs)
    - métricas: TP, FP, FN, TN, F1_Score, Precision, Recall, MCC, BER, etc.

Salida:
    artifacts/all_classifiers_results.csv
"""

from pathlib import Path
import pandas as pd


# Columnas mínimas que esperamos encontrar para considerar que un CSV es de resultados
RESULT_MIN_COLUMNS = {"dataset", "balancing_method", "scaler_type", "model_label"}


def infer_pipeline_from_path(path: Path) -> str:
    """
    Intenta inferir si el archivo pertenece a:
      - RUTA A (sin selección de características)
      - RUTA B (con selección de características / FS)

    Heurística sencilla basada en el nombre/ruta del archivo.
    Ajusta las condiciones si usas nombres diferentes.
    """
    path_str = str(path).lower()

    # Si la ruta contiene algo como "fs", "feature_selection", "03b_scaling_fs"...
    if "fs" in path_str or "feature_selection" in path_str or "03b_scaling_fs" in path_str:
        return "B_FS"               # RUTA B (con selección de características)
    else:
        return "A_BASELINE"         # RUTA A (sin FS)


def collect_results_from_artifacts(artifacts_root: Path) -> pd.DataFrame:
    """
    Recorre recursivamente artifacts_root en busca de CSV con columnas de resultados
    de clasificadores y los concatena en un solo DataFrame.
    """
    all_frames = []

    print(f"🔎 Buscando CSV de resultados en: {artifacts_root}")

    for csv_path in artifacts_root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"   ✗ No se pudo leer {csv_path}: {e}")
            continue

        # Verificamos si tiene las columnas mínimas para ser considerado "resultados de modelos"
        if not RESULT_MIN_COLUMNS.issubset(set(df.columns)):
            # Es probablemente un CSV de datos (preprocesamiento, balanceo, etc.)
            continue

        print(f"   ✓ Archivo de resultados detectado: {csv_path}")

        # Asegurar columna pipeline
        if "pipeline" not in df.columns:
            pipeline_label = infer_pipeline_from_path(csv_path)
            df["pipeline"] = pipeline_label

        # Agregar info de origen (útil para debug)
        df["source_csv"] = str(csv_path.relative_to(artifacts_root))

        all_frames.append(df)

    if not all_frames:
        print("⚠ No se encontraron CSV de resultados con las columnas esperadas.")
        return pd.DataFrame()

    all_results = pd.concat(all_frames, ignore_index=True)
    return all_results


def main():
    artifacts_root = Path("artifacts")

    all_results = collect_results_from_artifacts(artifacts_root)

    if all_results.empty:
        print("⚠ No se generó archivo agregado porque no se encontraron resultados.")
        return

    # Ordenar columnas un poquito (si existen)
    preferred_order = [
        "dataset",
        "pipeline",           # A_BASELINE / B_FS
        "balancing_method",
        "scaler_type",
        "model_label",
        "TP", "FP", "FN", "TN",
        "Precision", "Recall", "F1_Score",
        "MCC", "BER",
        "PR_AUC", "ROC_AUC",
        "score",
        "source_csv",
    ]

    # Reordenar las columnas que sí existan
    cols_existing = [c for c in preferred_order if c in all_results.columns]
    cols_remaining = [c for c in all_results.columns if c not in cols_existing]
    all_results = all_results[cols_existing + cols_remaining]

    out_path = artifacts_root / "all_classifiers_results.csv"
    all_results.to_csv(out_path, index=False)

    print("\n================================================================================")
    print(f"🎉 Archivo agregado generado en: {out_path}")
    print(f"   Filas totales: {len(all_results)}")
    print("================================================================================")


if __name__ == "__main__":
    main()