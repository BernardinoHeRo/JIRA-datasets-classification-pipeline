# src/hyperparameter_tuning.py

import json
import pickle
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    PHASE_03_SCALING_DIR,
    PHASE_03_FS_SCALING_DIR,
    PHASE_04_HYPERPARAMETER_DIR,
    PHASE_04_HYPERPARAMETER_FS_DIR,
)

# =====================================================================
# CARGA DE DATOS
# =====================================================================

def load_scaled_data(dataset_name: str, method: str, scaler_type: str, use_fs: bool = False):
    base_root = PHASE_03_FS_SCALING_DIR if use_fs else PHASE_03_SCALING_DIR
    base_path = base_root / method / dataset_name / scaler_type

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


# =====================================================================
# MODELOS + GRIDS (NO CAMBIAR: basado en literatura)
# =====================================================================

def get_hyperparameter_grids():
    return {
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


def get_models():
    return {
        "svm": SVC(random_state=42, cache_size=2000, max_iter=10000),
        "naive_bayes_gaussian": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }


def _count_param_combinations(param_grid: dict) -> int:
    if not param_grid:
        return 0
    return int(np.prod([len(v) for v in param_grid.values()]))


# =====================================================================
# GRIDSEARCH + EVALUACIÓN
# =====================================================================

def perform_grid_search(model, param_grid, X_train, y_train, cv_splits: int = 10, model_name: str = ""):
    # CV explícito, estratificado y con shuffle (reproducible)
    cv_obj = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # SVM suele ir mejor sin paralelismo interno si ya paralelizas por procesos
    n_jobs_inner = 1 if model_name == "svm" else -1

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_obj,
        scoring="recall",          # prioriza capturar defectos
        n_jobs=n_jobs_inner,
        verbose=0,
        return_train_score=False,
        error_score="raise",
    )
    gs.fit(X_train, y_train)
    return gs


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }


# =====================================================================
# CACHE + GUARDADO
# =====================================================================

def check_cache(cache_file: Path):
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for model_name, model_results in results.items():
        if model_results is None:
            continue

        best_params_file = output_dir / f"{model_name}_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(
                {
                    "best_params": model_results["best_params"],
                    "best_cv_score": model_results["best_cv_score"],
                    "test_metrics": model_results["test_metrics"],
                    "timestamp": model_results.get("timestamp", ""),
                    "cv_folds": model_results.get("cv_folds", None),
                    "n_combinations_tested": model_results.get("n_combinations_tested", None),
                    "total_fits": model_results.get("total_fits", None),
                    "time_seconds": model_results.get("time_seconds", None),
                    "dataset_info": model_results.get("dataset_info", None),
                },
                f,
                indent=2,
            )

        if "best_estimator" in model_results:
            model_file = output_dir / f"{model_name}_best_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model_results["best_estimator"], f)

        saved_count += 1

    summary_file = output_dir / "hyperparameter_search_summary.json"
    summary_data = {
        k: {kk: vv for kk, vv in v.items() if kk != "best_estimator"} if v else None
        for k, v in results.items()
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f" Guardado: {saved_count}/{len(results)} modelos → {output_dir}")


# =====================================================================
# PROCESO DE UN MODELO (PARA PARALELIZAR)
# =====================================================================

def process_single_model(args):
    model_name, model, param_grid, X_train, X_test, y_train, y_test, output_dir, use_cache, cv = args

    if use_cache:
        cache_file = output_dir / f"{model_name}_best_params.json"
        cached = check_cache(cache_file)
        if cached is not None:
            return model_name, cached

    try:
        start_time = datetime.now()
        combos = _count_param_combinations(param_grid)

        gs = perform_grid_search(model, param_grid, X_train, y_train, cv_splits=cv, model_name=model_name)
        test_metrics = evaluate_model(gs.best_estimator_, X_test, y_test)

        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            "best_params": gs.best_params_,
            "best_cv_score": float(gs.best_score_),  # recall
            "test_metrics": test_metrics,
            "n_combinations_tested": len(gs.cv_results_["params"]),
            "cv_folds": gs.n_splits_,
            "total_fits": len(gs.cv_results_["params"]) * gs.n_splits_,
            "time_seconds": elapsed,
            "dataset_info": {"n_samples": int(len(X_train)), "n_features": int(X_train.shape[1])},
            "timestamp": datetime.now().isoformat(),
            "best_estimator": gs.best_estimator_,
        }

        # salida mínima (1 línea)
        print(
            f"   ✓ {model_name:<20} | combos={combos:<4} | cv={cv} "
            f"| bestCV(recall)={result['best_cv_score']:.4f} "
            f"| testF1={test_metrics['f1_score']:.4f} | {elapsed:.1f}s"
        )

        return model_name, result

    except Exception as e:
        print(f"   ✗ {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return model_name, None


# =====================================================================
# NÚCLEO: UNA CONFIG (DATASET + MÉTODO + SCALER) PARA RUTA A/B
# =====================================================================

def tune_hyperparameters_for_dataset(
    dataset_name: str,
    method: str,
    scaler_type: str,
    use_fs: bool = False,
    use_parallel: bool = True,
    use_cache: bool = True,
    cv: int = 10,  # 10-fold
):
    ruta_tag = "FS" if use_fs else "NOFS"

    print(f"\n{'='*78}")
    print(f"[GRIDSEARCH] {ruta_tag} | {dataset_name} | {method} | {scaler_type} | cv={cv} | scoring=recall")
    print(f"{'='*78}")

    try:
        X_train, X_test, y_train, y_test = load_scaled_data(dataset_name, method, scaler_type, use_fs=use_fs)
    except FileNotFoundError as e:
        print(f"   ⚠ Datos no encontrados: {e}")
        return None

    base_output_root = PHASE_04_HYPERPARAMETER_FS_DIR if use_fs else PHASE_04_HYPERPARAMETER_DIR
    output_dir = base_output_root / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)

    models = get_models()
    param_grids = get_hyperparameter_grids()

    results = {}

    if use_parallel:
        tasks = [
            (name, model, param_grids[name], X_train, X_test, y_train, y_test, output_dir, use_cache, cv)
            for name, model in models.items()
        ]

        with ProcessPoolExecutor(max_workers=min(3, len(models))) as executor:
            futures = {executor.submit(process_single_model, t): t[0] for t in tasks}
            for future in as_completed(futures):
                model_name, res = future.result()
                results[model_name] = res
    else:
        for name, model in models.items():
            _, res = process_single_model(
                (name, model, param_grids[name], X_train, X_test, y_train, y_test, output_dir, use_cache, cv)
            )
            results[name] = res

    save_results(results, output_dir)
    return results


# =====================================================================
# WRAPPERS (UNA RUTA)
# =====================================================================

def tune_hyperparameters(dataset_name: str, use_fs: bool = False, cv: int = 10):
    from src.config import BALANCING_METHODS, SCALING_TYPES

    ruta_tag = "FS" if use_fs else "NOFS"
    print(f"\n{'='*80}")
    print(f"[HYPERPARAMETER TUNING] {ruta_tag} | dataset={dataset_name} | cv={cv}")
    print(f"{'='*80}")

    all_results = {}
    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            key = f"{method}_{scaler_type}"
            all_results[key] = tune_hyperparameters_for_dataset(
                dataset_name,
                method,
                scaler_type,
                use_fs=use_fs,
                use_parallel=True,
                use_cache=True,
                cv=cv,
            )
    return all_results


def tune_hyperparameters_fs(dataset_name: str, cv: int = 10):
    return tune_hyperparameters(dataset_name, use_fs=True, cv=cv)


if __name__ == "__main__":
    from src.config import DATASETS
    if DATASETS:
        tune_hyperparameters(DATASETS[0], use_fs=False, cv=10)
        # tune_hyperparameters(DATASETS[0], use_fs=True, cv=10)