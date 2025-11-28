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

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()

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
            cache_size=2000,  # 2GB cache para M4 Pro (tiene RAM de sobra)
            max_iter=10000    # Límite de iteraciones para evitar que se cuelgue
        ),
        'naive_bayes_gaussian': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    return models


def perform_grid_search(model, param_grid, X_train, y_train, cv=3, model_name=''):
    """
    Realiza Grid Search con validación cruzada
    OPTIMIZADO PARA M4 PRO: 
    - cv=3 (balance velocidad/confiabilidad)
    - Ajustes específicos por modelo para maximizar velocidad
    """
    # Ajustar n_jobs según el modelo para optimizar uso de CPU en M4 Pro
    if model_name == 'svm':
        # SVM es CPU-intensive, limitar workers internos
        n_jobs_inner = 1  # GridSearch usa -1, pero SVM usa 1 thread
    else:
        # Decision Tree y Naive Bayes son más rápidos, pueden usar más threads
        n_jobs_inner = -1
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,  # 3-fold: 33% más rápido que 5-fold
        scoring='f1',  # Optimiza directamente para F1
        n_jobs=-1,  # GridSearch paraleliza las combinaciones
        verbose=0,  # Sin output = más rápido
        return_train_score=False,  # No calcular train score = más rápido
        error_score='raise'  # Detectar errores inmediatamente
    )

    grid_search.fit(X_train, y_train)
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

    # Guardar mejores hiperparámetros para cada modelo
    for model_name, model_results in results.items():
        if model_results is not None:
            # Mejores hiperparámetros
            best_params_file = output_dir / f"{model_name}_best_params.json"
            with open(best_params_file, 'w') as f:
                json.dump({
                    'best_params': model_results['best_params'],
                    'best_cv_score': model_results['best_cv_score'],
                    'test_metrics': model_results['test_metrics'],
                    'timestamp': model_results.get('timestamp', '')
                }, f, indent=2)
            
            # Guardar modelo entrenado
            if 'best_estimator' in model_results:
                model_file = output_dir / f"{model_name}_best_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_results['best_estimator'], f)

    # Guardar resumen de todos los resultados
    summary_file = output_dir / "hyperparameter_search_summary.json"
    summary_data = {
        k: {kk: vv for kk, vv in v.items() if kk != 'best_estimator'} 
        if v else None 
        for k, v in results.items()
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)


def process_single_model(args):
    """
    Procesa un solo modelo - diseñado para paralelización
    Mantiene GridSearchCV pero optimiza el proceso
    OPTIMIZADO PARA M4 PRO con grids completos de literatura
    """
    model_name, model, param_grid, X_train, X_test, y_train, y_test, output_dir, use_cache = args
    
    # Verificar caché
    if use_cache:
        cache_file = output_dir / f"{model_name}_best_params.json"
        cached = check_cache(cache_file)
        if cached is not None:
            print(f"    ✓ {model_name}: usando resultados en caché")
            return model_name, cached
    
    try:
        print(f"    → {model_name}: iniciando GridSearch...")
        start_time = datetime.now()
        
        # Contar combinaciones reales (SVM ignora gamma con kernel linear)
        n_combinations = len(param_grid[list(param_grid.keys())[0]])
        for key in list(param_grid.keys())[1:]:
            n_combinations *= len(param_grid[key])
        
        # Información del dataset
        n_samples = len(X_train)
        n_features = X_train.shape[1]
        print(f"       Dataset: {n_samples} muestras, {n_features} features")
        
        if model_name == 'svm':
            # SVM: kernel='linear' ignora gamma, reduciendo combinaciones efectivas
            effective_combinations = (
                len(param_grid['C']) * 1 +  # linear kernel (gamma ignorado)
                len(param_grid['C']) * len(param_grid['gamma'])  # rbf kernel
            )
            print(f"       {n_combinations} combinaciones teóricas → ~{effective_combinations} efectivas")
            print(f"       Estimado: {effective_combinations * 3 * 0.5:.0f}-{effective_combinations * 3:.0f}s (~{effective_combinations * 3 / 60:.1f} min)")
        else:
            total_fits = n_combinations * 3  # 3-fold CV
            print(f"       {n_combinations} combinaciones × 3 folds = {total_fits} entrenamientos")
            if model_name == 'decision_tree':
                print(f"       Estimado: {total_fits * 0.1:.0f}-{total_fits * 0.3:.0f}s (~{total_fits * 0.2 / 60:.1f} min)")
            else:  # naive_bayes
                print(f"       Estimado: {total_fits * 0.05:.0f}-{total_fits * 0.1:.0f}s (<1 min)")
        
        # Grid Search con optimizaciones específicas
        grid_search = perform_grid_search(
            model, param_grid, X_train, y_train, cv=3, model_name=model_name
        )

        # Evaluar en test
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
            'best_estimator': grid_search.best_estimator_  # Para guardar después
        }
        
        print(f"    ✓ {model_name}: F1={test_metrics['f1_score']:.4f} | Tiempo real: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"       Mejor CV F1: {grid_search.best_score_:.4f} | Test F1: {test_metrics['f1_score']:.4f}")
        return model_name, results

    except Exception as e:
        print(f"    ✗ {model_name}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return model_name, None


def tune_hyperparameters_for_dataset(dataset_name, method, scaler_type, use_parallel=True, use_cache=True):
    """
    Realiza búsqueda de hiperparámetros para un dataset específico
    USANDO GRIDSEARCH (requerimiento del experimento)
    
    Args:
        use_parallel: Si True, procesa modelos en paralelo (MÁS RÁPIDO)
        use_cache: Si True, usa resultados previos si existen
    """
    print(f"\n→ {dataset_name} | {method} | {scaler_type}")

    # Cargar datos
    try:
        X_train, X_test, y_train, y_test = load_scaled_data(dataset_name, method, scaler_type)
    except FileNotFoundError as e:
        print(f"  ✗ Datos no encontrados")
        return None

    # Preparar output
    output_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener modelos y grids
    models = get_models()
    param_grids = get_hyperparameter_grids()

    results = {}

    if use_parallel:
        # MODO PARALELO: Procesa los 3 modelos simultáneamente
        tasks = [
            (name, model, param_grids[name], X_train, X_test, y_train, y_test, output_dir, use_cache)
            for name, model in models.items()
        ]
        
        # Usar máximo 3 workers (uno por modelo)
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_single_model, task): task[0] for task in tasks}
            
            for future in as_completed(futures):
                model_name, result = future.result()
                results[model_name] = result
    else:
        # MODO SECUENCIAL: Un modelo a la vez
        for model_name, model in models.items():
            _, result = process_single_model(
                (model_name, model, param_grids[model_name], 
                 X_train, X_test, y_train, y_test, output_dir, use_cache)
            )
            results[model_name] = result

    # Guardar resultados
    save_results(results, output_dir)

    return results


def tune_hyperparameters_batch(datasets, methods, scaler_types, max_parallel_configs=4):
    """
    Procesa múltiples configuraciones en paralelo
    OPTIMIZADO PARA M4 PRO: Procesa 4 configuraciones a la vez, cada una con 3 modelos en paralelo
    
    Args:
        max_parallel_configs: Configuraciones simultáneas (M4 Pro: usar 4-6)
    """
    from itertools import product
    
    # Generar todas las combinaciones
    all_configs = list(product(datasets, methods, scaler_types))
    total = len(all_configs)
    
    print(f"\n{'='*80}")
    print(f"GRIDSEARCH OPTIMIZADO PARA M4 PRO - {total} CONFIGURACIONES")
    print(f"Modo: {max_parallel_configs} configs simultáneas × 3 modelos paralelos")
    print(f"CPU disponibles: {max_parallel_configs * 3} workers totales")
    print(f"{'='*80}\n")

    start_time = datetime.now()
    
    # Procesar en lotes con paralelización limitada
    with ProcessPoolExecutor(max_workers=max_parallel_configs) as executor:
        futures = []
        for dataset, method, scaler in all_configs:
            future = executor.submit(
                tune_hyperparameters_for_dataset,
                dataset, method, scaler, 
                use_parallel=True,  # Cada config usa paralelización interna
                use_cache=True
            )
            futures.append((future, dataset, method, scaler))
        
        # Recolectar resultados
        completed = 0
        for future, dataset, method, scaler in futures:
            try:
                result = future.result()
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
    """
    Realiza búsqueda de hiperparámetros para todas las combinaciones
    de un dataset (todos los métodos de balanceo y tipos de escalado)
    """
    from src.config import BALANCING_METHODS, SCALING_TYPES

    print(f"\n{'='*80}")
    print(f"BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH: {dataset_name}")
    print(f"{'='*80}")

    all_results = {}

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            combination_key = f"{method}_{scaler_type}"
            print(f"\n{'-'*60}")
            print(f"Procesando: {combination_key}")
            print(f"{'-'*60}")

            results = tune_hyperparameters_for_dataset(
                dataset_name, method, scaler_type,
                use_parallel=True,
                use_cache=True
            )
            all_results[combination_key] = results

    return all_results


if __name__ == "__main__":
    from src.config import DATASETS, BALANCING_METHODS, SCALING_TYPES

    # OPCIÓN 1: Procesar un dataset
    if len(DATASETS) > 0:
        tune_hyperparameters(DATASETS[0])

    # OPCIÓN 2: Procesar todo en paralelo (RECOMENDADO PARA M4 PRO)
    # tune_hyperparameters_batch(
    #     datasets=DATASETS,
    #     methods=BALANCING_METHODS,
    #     scaler_types=SCALING_TYPES,
    #     max_parallel_configs=5  # M4 Pro: puedes usar 4-6
    # )