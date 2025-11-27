# src/hyperparameter_tuning.py

# ============================
# BÚSQUEDA DE HIPERPARÁMETROS
# ============================
# Implementa GridSearchCV para optimizar hiperparámetros de:
# - SVM (Support Vector Machine)
# - Naive Bayes (Gaussian y Multinomial)
# - Decision Tree
#
# Para cada combinación de dataset, método de balanceo y tipo de escalado
# ============================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from src.config import PHASE_03_SCALING_DIR, PHASE_04_HYPERPARAMETER_DIR


def load_scaled_data(dataset_name, method, scaler_type):
    """
    Carga los datos escalados desde artifacts/03_scaling/

    Args:
        dataset_name: Nombre del dataset
        method: Método de balanceo (csbboost, hcbou, smote)
        scaler_type: Tipo de escalado (standard, robust)

    Returns:
        X_train, X_test, y_train, y_test: DataFrames/Series con los datos
    """
    base_path = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def get_hyperparameter_grids():
    """
    Define los grids de hiperparámetros para cada modelo

    Returns:
        dict: Diccionario con los grids para cada modelo
    """
    grids = {
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'naive_bayes_gaussian': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        'decision_tree': {
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy']
        }
    }
    return grids


def get_models():
    """
    Obtiene los modelos base para Grid Search

    Returns:
        dict: Diccionario con los modelos
    """
    models = {
        'svm': SVC(random_state=42),
        'naive_bayes_gaussian': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    return models


def perform_grid_search(model, param_grid, X_train, y_train, cv=5):
    """
    Realiza Grid Search con validación cruzada

    Args:
        model: Modelo de sklearn
        param_grid: Grid de hiperparámetros
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        cv: Número de folds para validación cruzada

    Returns:
        grid_search: Objeto GridSearchCV ajustado
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Usar todos los procesadores disponibles
        verbose=1
    )

    print(f"    Iniciando Grid Search con {cv}-fold CV...")
    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de test

    Args:
        model: Modelo entrenado
        X_test: Datos de test
        y_test: Etiquetas de test

    Returns:
        dict: Métricas de evaluación
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    return metrics


def save_results(results, output_dir):
    """
    Guarda los resultados de la búsqueda de hiperparámetros

    Args:
        results: Diccionario con los resultados
        output_dir: Directorio de salida
    """
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
                    'test_metrics': model_results['test_metrics']
                }, f, indent=2)

    # Guardar resumen de todos los resultados
    summary_file = output_dir / "hyperparameter_search_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)


def tune_hyperparameters_for_dataset(dataset_name, method, scaler_type):
    """
    Realiza búsqueda de hiperparámetros para un dataset específico

    Args:
        dataset_name: Nombre del dataset
        method: Método de balanceo
        scaler_type: Tipo de escalado
    """
    print(f"\n--- Tuning hiperparámetros: {dataset_name} | {method} | {scaler_type} ---")

    # Cargar datos
    try:
        X_train, X_test, y_train, y_test = load_scaled_data(dataset_name, method, scaler_type)
    except FileNotFoundError as e:
        print(f"  ERROR: No se encontraron los datos escalados: {e}")
        return None

    # Obtener modelos y grids
    models = get_models()
    param_grids = get_hyperparameter_grids()

    results = {}

    # Verificar si los datos son apropiados para MultinomialNB (requiere valores no negativos)
    has_negative_values = (X_train < 0).any().any()

    for model_name, model in models.items():
        print(f"\n  === {model_name.upper()} ===")

        try:
            # Realizar Grid Search
            grid_search = perform_grid_search(
                model,
                param_grids[model_name],
                X_train,
                y_train
            )

            print(f"    Mejores hiperparámetros: {grid_search.best_params_}")
            print(f"    Mejor CV score (Accuracy): {grid_search.best_score_:.4f}")

            # Evaluar en test set
            test_metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test)
            print(f"    Test F1-score: {test_metrics['f1_score']:.4f}")
            print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")

            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'test_metrics': test_metrics,
                'cv_results_mean': grid_search.cv_results_['mean_test_score'].tolist(),
                'cv_results_std': grid_search.cv_results_['std_test_score'].tolist()
            }

        except Exception as e:
            print(f"    ERROR en {model_name}: {e}")
            results[model_name] = None

    # Guardar resultados
    output_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    save_results(results, output_dir)

    return results


def tune_hyperparameters(dataset_name):
    """
    Realiza búsqueda de hiperparámetros para todas las combinaciones
    de un dataset (todos los métodos de balanceo y tipos de escalado)

    Args:
        dataset_name: Nombre del dataset a procesar
    """
    from src.config import BALANCING_METHODS, SCALING_TYPES

    print(f"\n{'='*80}")
    print(f"BÚSQUEDA DE HIPERPARÁMETROS PARA: {dataset_name}")
    print(f"{'='*80}")

    all_results = {}

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            combination_key = f"{method}_{scaler_type}"
            print(f"\n{'-'*60}")
            print(f"Procesando: {combination_key}")
            print(f"{'-'*60}")

            results = tune_hyperparameters_for_dataset(dataset_name, method, scaler_type)
            all_results[combination_key] = results

    return all_results


if __name__ == "__main__":
    # Ejemplo de uso: procesar un dataset específico
    from src.config import DATASETS

    # Procesar el primer dataset como ejemplo
    if DATASETS:
        tune_hyperparameters(DATASETS[0])
    else:
        print("No hay datasets definidos en config.py")