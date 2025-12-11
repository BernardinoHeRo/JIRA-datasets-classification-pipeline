# src/hyperparameter_tuning.py

# ============================================================
#  FASE 4: BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH
# ============================================================
# Objetivo:
#   - Cargar datos ya balanceados y escalados (FASE 2 y 3)
#   - Para cada combinación:
#         dataset × método de balanceo × tipo de escalado
#     buscar los mejores hiperparámetros de:
#         - SVM
#         - Gaussian Naive Bayes
#         - Árbol de decisión
#   - Usar GridSearchCV con validación cruzada estratificada,
#     optimizando una métrica enfocada en la clase defectuosa (Buggy).
#       → Usamos F2-score (β = 2), que da más peso al Recall.
#   - Evaluar en el conjunto de prueba y guardar:
#         - mejores hiperparámetros
#         - métricas de test
#         - matriz de confusión (filas=clase real, columnas=predicha)
#         - modelo entrenado con los mejores parámetros
# ============================================================

import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    fbeta_score,          # para F2-score
    make_scorer,          # para definir un scorer personalizado
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import PHASE_03_SCALING_DIR, PHASE_04_HYPERPARAMETER_DIR


# ============================================================
# 0. SCORER PERSONALIZADO (F2 para clase defectuosa)
# ============================================================

# Asumimos que la clase "defectuosa" (Buggy) está codificada como 1
BUGGY_LABEL = 1

# Usamos F2-score: β = 2 → prioriza el Recall frente a la Precisión
F_BETA = 2.0

BUGGY_F2_SCORER = make_scorer(
    fbeta_score,
    beta=F_BETA,
    pos_label=BUGGY_LABEL,
    average="binary",
    zero_division=0,
)


# ============================================================
# 1. UTILIDADES DE CARGA DE DATOS
# ============================================================

def load_scaled_data(dataset_name: str, method: str, scaler_type: str):
    """
    Carga los datos escalados desde:
        artifacts/03_scaling/<method>/<dataset>/<scaler_type>/

    Se espera que esa carpeta contenga:
        - X_train_scaled.csv
        - X_test_scaled.csv
        - y_train.csv
        - y_test.csv
    """
    base_path = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()

    # 🆕 Aseguramos que y siga en 0/1, por seguridad
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    unique_train = sorted(y_train.unique().tolist())
    unique_test = sorted(y_test.unique().tolist())
    print(f"       📌 Etiquetas únicas y_train: {unique_train}")
    print(f"       📌 Etiquetas únicas y_test : {unique_test}")

    if not set(unique_train).issubset({0, 1}) or not set(unique_test).issubset({0, 1}):
        raise ValueError(
            f"       ❌ ERROR: Las etiquetas no están en {{0,1}} después del escalado/balanceo.\n"
            f"          y_train: {unique_train} | y_test: {unique_test}\n"
            f"          Verifica el pipeline de balanceo/escala para que conserve RealBug en 0/1."
        )

    print(f"       ✓ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"       ✓ Target codificado en 0/1 (Buggy = {BUGGY_LABEL})")
    return X_train, X_test, y_train, y_test


# ============================================================
# 2. DEFINICIÓN DE MODELOS Y GRIDS DE HIPERPARÁMETROS
# ============================================================

def get_hyperparameter_grids():
    """
    Define los grids de hiperparámetros según literatura científica.

    Referencias:
    - SVM: Hsu, Chang & Lin (2010); Feurer et al. (2015)
    - Naive Bayes: Kuhn & Johnson (2013); Scikit-Learn (2023)
    - Decision Tree: Hastie, Tibshirani & Friedman (2009); Kuhn & Johnson (2013)
    """
    grids = {
        "svm": {
            "C": [0.01, 0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "rbf"],
            "gamma": [0.001, 0.01, 0.1, 1, "scale", "auto"],
        },
        "naive_bayes_gaussian": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
        },
        "decision_tree": {
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
        },
    }
    return grids


def get_models():
    """
    Devuelve los modelos base que serán ajustados mediante GridSearchCV.
    """
    models = {
        "svm": SVC(
            random_state=42,
            cache_size=2000,
            max_iter=10000,
        ),
        "naive_bayes_gaussian": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }
    return models


# ============================================================
# 3. GRID SEARCH Y EVALUACIÓN
# ============================================================

def perform_grid_search(model, param_grid, X_train, y_train, cv: int = 10):
    """
    Ejecuta GridSearchCV usando validación cruzada estratificada.

    - Métrica de optimización:
        BUGGY_F2_SCORER → F2-score (β = 2) para la clase positiva (defectuosa = 1),
        priorizando Recall sin olvidar la Precisión.
    - CV: StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    """
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=BUGGY_F2_SCORER,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.

    Retorna:
        metrics: dict con accuracy, precision, recall, f1.
        cm: matriz de confusión (numpy.ndarray), filas=clase real, columnas=predicha.
        labels: arreglo con las etiquetas en el orden de la matriz.
    """
    y_pred = model.predict(X_test)

    # Métricas básicas (clase positiva asumida = 1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        # 🆕 también guardamos F2 en test, por consistencia
        "f2_score": float(
            fbeta_score(
                y_test,
                y_pred,
                beta=F_BETA,
                pos_label=BUGGY_LABEL,
                average="binary",
                zero_division=0,
            )
        ),
    }

    # Matriz de confusión (filas = reales, columnas = predichas)
    labels = np.unique(np.concatenate([np.array(y_test), np.array(y_pred)]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return metrics, cm, labels


# ============================================================
# 4. CACHÉ Y GUARDADO DE RESULTADOS
# ============================================================

def check_cache(cache_file: Path):
    """Devuelve el contenido del caché si existe y es válido, o None en caso contrario."""
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_results(results: dict, output_dir: Path):
    """
    Guarda los resultados de la búsqueda de hiperparámetros.

    Por cada modelo se guarda:
        - <modelo>_best_params.json
        - <modelo>_best_model.pkl      (si está disponible)
    Además, se genera:
        - hyperparameter_search_summary.json

    Es tolerante a resultados viejos que no tengan
    'confusion_matrix' ni 'labels' (caché previo).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n       💾 Guardando resultados en: {output_dir}")

    saved_count = 0

    for model_name, model_results in results.items():
        if model_results is None:
            continue

        # Soportar tanto resultados nuevos como caché viejo
        cm = model_results.get("confusion_matrix", None)
        labels = model_results.get("labels", None)

        payload = {
            "best_params": model_results.get("best_params"),
            "best_cv_score": model_results.get("best_cv_score"),
            "test_metrics": model_results.get("test_metrics"),
            "timestamp": model_results.get("timestamp", ""),
        }

        # Solo agregamos matriz de confusión y etiquetas si existen
        if cm is not None:
            payload["confusion_matrix"] = cm
        if labels is not None:
            payload["labels"] = labels

        # Guardar JSON por modelo
        best_params_file = output_dir / f"{model_name}_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(payload, f, indent=2)

        # Guardar modelo solo si está disponible (no habrá en caché viejo)
        if "best_estimator" in model_results:
            model_file = output_dir / f"{model_name}_best_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model_results["best_estimator"], f)

        saved_count += 1
        print(f"          ✓ {model_name}: guardado")

    # Resumen general (sin incluir el estimador para que sea ligero)
    summary_file = output_dir / "hyperparameter_search_summary.json"
    summary_data = {}

    for k, v in results.items():
        if v is None:
            summary_data[k] = None
            continue

        copy_v = {
            kk: vv
            for kk, vv in v.items()
            if kk not in ("best_estimator",)
        }
        summary_data[k] = copy_v

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"       ✓ Resumen guardado: {saved_count}/{len(results)} modelos")


# ============================================================
# 5. PROCESAMIENTO DE UN MODELO (PARALELIZABLE)
# ============================================================

def process_single_model(args):
    """
    Procesa un solo modelo para una configuración dada.

    Parámetros (empaquetados en args):
        model_name : str
        model      : estimador sklearn
        param_grid : dict
        X_train, X_test, y_train, y_test
        output_dir : Path
        use_cache  : bool
    """
    (
        model_name,
        model,
        param_grid,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir,
        use_cache,
    ) = args

    cache_file = output_dir / f"{model_name}_best_params.json"

    # 1) Si hay caché válido, lo usamos directamente
    if use_cache:
        cached = check_cache(cache_file)
        if cached is not None:
            print(f"    ✓ {model_name}: usando resultados en caché")
            return model_name, cached

    try:
        print(f"    → {model_name}: iniciando GridSearch...")
        start_time = datetime.now()

        grid_search = perform_grid_search(
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            cv=10,
        )

        # 2) Evaluación en test
        test_metrics, cm, labels = evaluate_model(
            grid_search.best_estimator_, X_test, y_test
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        # 3) Empaquetar resultados
        results = {
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "test_metrics": test_metrics,
            "confusion_matrix": cm.tolist(),
            "labels": labels.tolist(),
            "n_combinations_tested": len(grid_search.cv_results_["params"]),
            "cv_folds": grid_search.n_splits_,
            "total_fits": len(grid_search.cv_results_["params"])
            * grid_search.n_splits_,
            "time_seconds": elapsed,
            "dataset_info": {
                "n_samples": int(len(X_train) + len(X_test)),
                "n_features": int(X_train.shape[1]),
            },
            "timestamp": datetime.now().isoformat(),
            "best_estimator": grid_search.best_estimator_,
        }

        # 4) Salida por consola resumida
        print(
            f"    ✓ {model_name}: "
            f"F1_test={test_metrics['f1_score']:.4f} | "
            f"F2_test={test_metrics['f2_score']:.4f} | "
            f"Recall_test={test_metrics['recall']:.4f} | "
            f"Tiempo={elapsed:.1f}s"
        )
        print(f"       Mejor F2 (CV) (β=2) = {grid_search.best_score_:.4f}")

        # 5) Imprimir matriz de confusión
        print("       Matriz de confusión (filas=clase real, columnas=predicha):")
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        print(df_cm)

        # 6) Imprimir mejores hiperparámetros
        print(f"       Mejores hiperparámetros ({model_name}):")
        for param, value in grid_search.best_params_.items():
            print(f"         - {param}: {value}")

        return model_name, results

    except Exception as e:
        print(f"    ✗ {model_name}: ERROR - {str(e)}")
        import traceback

        traceback.print_exc()
        return model_name, None


# ============================================================
# 6. PROCESAMIENTO POR DATASET / CONFIGURACIÓN
# ============================================================

def tune_hyperparameters_for_dataset(
    dataset_name: str,
    method: str,
    scaler_type: str,
    use_parallel: bool = True,
    use_cache: bool = True,
):
    """
    Ejecuta la búsqueda de hiperparámetros para un dataset,
    un método de balanceo y un tipo de escalado específicos.
    """
    print(f"\n{'='*70}")
    print(f"🔧 CONFIGURACIÓN: dataset={dataset_name} | method={method} | scaler={scaler_type}")
    print(f"{'='*70}")

    # 1) Carga de datos escalados
    try:
        X_train, X_test, y_train, y_test = load_scaled_data(
            dataset_name, method, scaler_type
        )
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: Datos no encontrados - {e}")
        return None

    # 2) Preparar carpeta de salida
    output_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"       📁 Directorio de salida: {output_dir}")

    # 3) Modelos y grids
    models = get_models()
    param_grids = get_hyperparameter_grids()

    print(f"\n       Modelos a procesar: {list(models.keys())}")
    print(f"       Modo: {'paralelo' if use_parallel else 'secuencial'}")
    print(f"       Caché: {'activado' if use_cache else 'desactivado'}")

    results = {}

    # 4) Procesamiento paralelo o secuencial de modelos
    if use_parallel:
        tasks = [
            (
                name,
                model,
                param_grids[name],
                X_train,
                X_test,
                y_train,
                y_test,
                output_dir,
                use_cache,
            )
            for name, model in models.items()
        ]

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_single_model, task): task[0] for task in tasks}
            completed = 0
            for future in as_completed(futures):
                model_name, result = future.result()
                results[model_name] = result
                completed += 1
                print(f"       Progreso modelos: {completed}/{len(models)} completados")
    else:
        for idx, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n       [{idx}/{len(models)}] Procesando modelo: {model_name}")
            _, result = process_single_model(
                (
                    model_name,
                    model,
                    param_grids[model_name],
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    output_dir,
                    use_cache,
                )
            )
            results[model_name] = result

    # 5) Guardar resultados en disco
    save_results(results, output_dir)

    print(f"\n{'='*70}")
    print(f"✅ CONFIGURACIÓN COMPLETADA: {dataset_name} | {method} | {scaler_type}")
    print(f"{'='*70}\n")

    return results


# ============================================================
# 7. BATCH DE CONFIGURACIONES (VARIOS DATASETS / MÉTODOS / ESCALERS)
# ============================================================

def tune_hyperparameters_batch(datasets, methods, scaler_types, max_parallel_configs: int = 4):
    """
    Procesa múltiples configuraciones en paralelo:

        datasets × methods × scaler_types

    max_parallel_configs controla cuántas configuraciones
    se procesan en paralelo (cada una con hasta 3 modelos en paralelo).
    """
    from itertools import product

    all_configs = list(product(datasets, methods, scaler_types))
    total = len(all_configs)

    print(f"\n{'='*80}")
    print(f"GRIDSEARCH OPTIMIZADO - {total} CONFIGURACIONES")
    print(f"Modo: {max_parallel_configs} configs simultáneas")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    with ProcessPoolExecutor(max_workers=max_parallel_configs) as executor:
        futures = []
        for dataset, method, scaler in all_configs:
            future = executor.submit(
                tune_hyperparameters_for_dataset,
                dataset,
                method,
                scaler,
                use_parallel=True,
                use_cache=True,
            )
            futures.append((future, dataset, method, scaler))

        completed = 0
        for future, dataset, method, scaler in futures:
            try:
                future.result()
                completed += 1
                print(f"[{completed}/{total}] ✓ {dataset} | {method} | {scaler} completado")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{total}] ✗ {dataset} | {method} | {scaler}: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*80}")
    print(f"COMPLETADO en {elapsed/60:.1f} minutos")
    print(f"{'='*80}\n")


# ============================================================
# 8. FUNCIÓN PRINCIPAL POR DATASET
# ============================================================

def tune_hyperparameters(dataset_name: str):
    """
    Ejecuta la búsqueda de hiperparámetros para TODAS las combinaciones de:
        - Métodos de balanceo definidos en src.config.BALANCING_METHODS
        - Tipos de escalado definidos en src.config.SCALING_TYPES
    sobre un único dataset.
    """
    from src.config import BALANCING_METHODS, SCALING_TYPES

    print(f"\n{'='*80}")
    print(f"🎯 BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'='*80}")

    total_configs = len(BALANCING_METHODS) * len(SCALING_TYPES)
    print(f"\n📋 Total de configuraciones: {total_configs}")
    print(f"   • Métodos de balanceo: {BALANCING_METHODS}")
    print(f"   • Tipos de escalado:   {SCALING_TYPES}")
    print(f"   • Modelos por config:  3\n")

    all_results = {}
    config_num = 0

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            config_num += 1
            combination_key = f"{method}_{scaler_type}"

            print(f"\n{'*'*80}")
            print(f"[{config_num}/{total_configs}] CONFIGURACIÓN: {combination_key}")
            print(f"{'*'*80}")

            results = tune_hyperparameters_for_dataset(
                dataset_name,
                method,
                scaler_type,
                use_parallel=True,
                use_cache=True,
            )
            all_results[combination_key] = results

            print(f"\n✅ Configuración {config_num}/{total_configs} completada: {combination_key}")

    print(f"\n{'='*80}")
    print(f"🎉 TODAS LAS CONFIGURACIONES COMPLETADAS PARA: {dataset_name}")
    print(f"{'='*80}\n")

    return all_results


# ============================================================
# 9. PUNTO DE ENTRADA (opcional)
# ============================================================

if __name__ == "__main__":
    from src.config import DATASETS

    if len(DATASETS) > 0:
        tune_hyperparameters(DATASETS[0])