# ============================
# BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH (OPTIMIZADO)
# ============================
# Implementa GridSearchCV optimizado para velocidad sin perder exhaustividad
# Optimizaciones: paralelización, caché, CV reducido, grids más pequeños
# ============================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from datetime import datetime

from src.config import PHASE_03_SCALING_DIR, PHASE_04_HYPERPARAMETER_DIR


def load_scaled_data(dataset_name, method, scaler_type):
    """Carga los datos escalados desde artifacts/03_scaling/"""
    base_path = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type
    
    print(f"       📂 Cargando datos desde: {base_path}")

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()
    
    print(f"       ✓ Datos cargados: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test


def get_hyperparameter_grids():
    """
    Define los grids de hiperparámetros según literatura científica
    
    Referencias:
    - SVM: Hsu, Chang & Lin (2010); Feurer et al. (2015)
    - Naive Bayes: Kuhn & Johnson (2013); Scikit-Learn (2023)
    - Decision Tree: Hastie, Tibshirani & Friedman (2009); Kuhn & Johnson (2013)
    """
    grids = {
        'svm': {
            'C': [0.01, 0.1, 1, 10, 100, 1000],  # 6 valores (Hsu, Chang & Lin, 2010)
            'kernel': ['linear', 'rbf'],  # 2 valores (Scikit-Learn, 2023)
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']  # 6 valores (Hsu et al., 2010; Feurer et al., 2015)
            # Total: 6×2×6 = 72 combinaciones
            # Con kernel='linear', gamma se ignora → ~42 combinaciones efectivas
        },
        'naive_bayes_gaussian': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]  # 6 valores (Kuhn & Johnson, 2013)
            # Total: 6 combinaciones
        },
        'decision_tree': {
            'max_depth': [3, 5, 10, 20, None],  # 5 valores (Hastie, Tibshirani & Friedman, 2009)
            'min_samples_split': [2, 5, 10, 20],  # 4 valores (Kuhn & Johnson, 2013)
            'min_samples_leaf': [1, 2, 4, 8],  # 4 valores (Kuhn & Johnson, 2013)
            'criterion': ['gini', 'entropy']  # 2 valores (Scikit-Learn, 2023)
            # Total: 5×4×4×2 = 160 combinaciones
        }
    }
    return grids


def get_models():
    """Obtiene los modelos base optimizados para M4 Pro"""
    models = {
        'svm': SVC(
            random_state=42, 
            cache_size=2000,
            max_iter=10000
        ),
        'naive_bayes_gaussian': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    return models


def perform_grid_search(model, param_grid, X_train, y_train, cv=10, model_name=''):
    """Realiza Grid Search con validación cruzada"""
    if model_name == 'svm':
        n_jobs_inner = 1
    else:
        n_jobs_inner = -1

    print(f"       🔍 Configurando GridSearchCV...")
    print(f"          - CV folds: {cv}")
    print(f"          - Scoring: f1")
    print(f"          - Paralelización: n_jobs=-1")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
        error_score='raise'
    )

    print(f"       ⚙️  Ejecutando Grid Search...")
    grid_search.fit(X_train, y_train)
    print(f"       ✓ Grid Search completado")

    return grid_search


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo en el conjunto de test"""
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
    }

    return metrics


def check_cache(cache_file):
    """Verifica si ya existe un resultado en caché"""
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_results(results, output_dir):
    """Guarda los resultados de la búsqueda de hiperparámetros"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n       💾 Guardando resultados en: {output_dir}")

    saved_count = 0
    for model_name, model_results in results.items():
        if model_results is not None:
            best_params_file = output_dir / f"{model_name}_best_params.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    'best_params': model_results['best_params'],
                    'best_cv_score': model_results['best_cv_score'],
                    'test_metrics': model_results['test_metrics'],
                    'timestamp': model_results.get('timestamp', '')
                }, f, indent=2)
            
            if 'best_estimator' in model_results:
                model_file = output_dir / f"{model_name}_best_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_results['best_estimator'], f)
            
            saved_count += 1
            print(f"          ✓ {model_name}: guardado")

    summary_file = output_dir / "hyperparameter_search_summary.json"
    summary_data = {
        k: {kk: vv for kk, vv in v.items() if kk != 'best_estimator'} 
        if v else None 
        for k, v in results.items()
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"       ✓ Resumen guardado: {saved_count}/{len(results)} modelos")


def process_single_model(args):
    """Procesa un solo modelo - diseñado para paralelización"""
    model_name, model, param_grid, X_train, X_test, y_train, y_test, output_dir, use_cache = args
    
    if use_cache:
        cache_file = output_dir / f"{model_name}_best_params.json"
        cached = check_cache(cache_file)
        if cached is not None:
            print(f"    ✓ {model_name}: usando resultados en caché")
            return model_name, cached
    
    try:
        print(f"    → {model_name}: iniciando GridSearch...")
        start_time = datetime.now()
        
        n_combinations = len(param_grid[list(param_grid.keys())[0]])
        for key in list(param_grid.keys())[1:]:
            n_combinations *= len(param_grid[key])
        
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        print(f"       Dataset: {n_samples} muestras, {n_features} features")
        
        if model_name == 'svm':
            effective_combinations = (
                len(param_grid['C']) * 1 +
                len(param_grid['C']) * len(param_grid['gamma'])
            )
            print(f"       {n_combinations} combinaciones teóricas → ~{effective_combinations} efectivas")
        else:
            total_fits = n_combinations * 3
            print(f"       {n_combinations} combinaciones × 3 folds = {total_fits} entrenamientos")
        
        grid_search = perform_grid_search(
            model, param_grid, X_train, y_train, cv=3, model_name=model_name
        )

        test_metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'test_metrics': test_metrics,
            'n_combinations_tested': len(grid_search.cv_results_['params']),
            'cv_folds': grid_search.n_splits_,
            'total_fits': len(grid_search.cv_results_['params']) * grid_search.n_splits_,
            'time_seconds': elapsed,
            'dataset_info': {'n_samples': n_samples, 'n_features': n_features},
            'timestamp': datetime.now().isoformat(),
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"    ✓ {model_name}: F1={test_metrics['f1_score']:.4f} | Tiempo real: {elapsed:.1f}s")
        print(f"       Mejor CV F1: {grid_search.best_score_:.4f} | Test F1: {test_metrics['f1_score']:.4f}")

        # >>>> IMPRESIÓN DE MEJORES HIPERPARÁMETROS <<<<
        print(f"       👉 Mejores hiperparámetros para {model_name}:")
        for param, value in grid_search.best_params_.items():
            print(f"          - {param}: {value}")

        return model_name, results

    except Exception as e:
        print(f"    ✗ {model_name}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return model_name, None


def tune_hyperparameters_for_dataset(dataset_name, method, scaler_type, use_parallel=True, use_cache=True):
    """Realiza búsqueda de hiperparámetros para un dataset específico"""
    print(f"\n{'='*70}")
    print(f"🔧 CONFIGURACIÓN: {dataset_name} | {method} | {scaler_type}")
    print(f"{'='*70}")

    try:
        X_train, X_test, y_train, y_test = load_scaled_data(dataset_name, method, scaler_type)
    except FileNotFoundError as e:
        print(f"  ✗ ERROR: Datos no encontrados - {e}")
        return None

    output_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"       📁 Directorio de salida: {output_dir}")

    models = get_models()
    param_grids = get_hyperparameter_grids()
    
    print(f"\n       📋 Modelos a procesar: {list(models.keys())}")
    print(f"       {'⚡ Modo PARALELO' if use_parallel else '🔄 Modo SECUENCIAL'}")
    print(f"       {'💾 Caché ACTIVADO' if use_cache else '🚫 Caché DESACTIVADO'}")

    results = {}

    if use_parallel:
        print(f"\n       🚀 Iniciando procesamiento paralelo de {len(models)} modelos...")
        tasks = [
            (name, model, param_grids[name], X_train, X_test, y_train, y_test, output_dir, use_cache)
            for name, model in models.items()
        ]
        
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_single_model, task): task[0] for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                model_name, result = future.result()
                results[model_name] = result
                completed += 1
                print(f"       ✓ Progreso: {completed}/{len(models)} modelos completados")
    else:
        print(f"\n       🔄 Iniciando procesamiento secuencial de {len(models)} modelos...")
        for idx, (model_name, model) in enumerate(models.items(), 1):
            print(f"\n       [{idx}/{len(models)}] Procesando modelo: {model_name}")
            _, result = process_single_model(
                (model_name, model, param_grids[model_name], 
                 X_train, X_test, y_train, y_test, output_dir, use_cache)
            )
            results[model_name] = result

    save_results(results, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ CONFIGURACIÓN COMPLETADA: {dataset_name} | {method} | {scaler_type}")
    print(f"{'='*70}\n")

    return results


def tune_hyperparameters_batch(datasets, methods, scaler_types, max_parallel_configs=4):
    """Procesa múltiples configuraciones en paralelo"""
    from itertools import product
    
    all_configs = list(product(datasets, methods, scaler_types))
    total = len(all_configs)
    
    print(f"\n{'='*80}")
    print(f"GRIDSEARCH OPTIMIZADO PARA M4 PRO - {total} CONFIGURACIONES")
    print(f"Modo: {max_parallel_configs} configs simultáneas × 3 modelos paralelos")
    print(f"{'='*80}\n")

    start_time = datetime.now()
    
    with ProcessPoolExecutor(max_workers=max_parallel_configs) as executor:
        futures = []
        for dataset, method, scaler in all_configs:
            future = executor.submit(
                tune_hyperparameters_for_dataset,
                dataset, method, scaler,
                use_parallel=True,
                use_cache=True
            )
            futures.append((future, dataset, method, scaler))
        
        completed = 0
        for future, dataset, method, scaler in futures:
            try:
                future.result()
                completed += 1
                print(f"\n[{completed}/{total}] ✓ {dataset}/{method}/{scaler} completado")
            except Exception as e:
                completed += 1
                print(f"\n[{completed}/{total}] ✗ {dataset}/{method}/{scaler}: {e}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*80}")
    print(f"COMPLETADO en {elapsed/60:.1f} minutos")
    print(f"{'='*80}\n")


def tune_hyperparameters(dataset_name):
    """Procesa todas las configuraciones de balanceo y escalado"""
    from src.config import BALANCING_METHODS, SCALING_TYPES

    print(f"\n{'='*80}")
    print(f"🎯 BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH")
    print(f"📊 Dataset: {dataset_name}")
    print(f"{'='*80}")

    total_configs = len(BALANCING_METHODS) * len(SCALING_TYPES)
    print(f"\n📋 Total de configuraciones a procesar: {total_configs}")
    print(f"   • Métodos de balanceo: {BALANCING_METHODS}")
    print(f"   • Tipos de escalado: {SCALING_TYPES}")
    print(f"   • Modelos por config: 3")

    all_results = {}
    config_num = 0

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            config_num += 1
            combination_key = f"{method}_{scaler_type}"
            
            print(f"\n{'*'*80}")
            print(f"[{config_num}/{total_configs}] PROCESANDO CONFIGURACIÓN: {combination_key}")
            print(f"{'*'*80}")

            results = tune_hyperparameters_for_dataset(
                dataset_name, method, scaler_type,
                use_parallel=True,
                use_cache=True
            )
            all_results[combination_key] = results
            
            print(f"\n✅ Configuración {config_num}/{total_configs} completada: {combination_key}")

    print(f"\n{'='*80}")
    print(f"🎉 TODAS LAS CONFIGURACIONES COMPLETADAS PARA: {dataset_name}")
    print(f"{'='*80}\n")

    return all_results


if __name__ == "__main__":
    from src.config import DATASETS

    if len(DATASETS) > 0:
        tune_hyperparameters(DATASETS[0])