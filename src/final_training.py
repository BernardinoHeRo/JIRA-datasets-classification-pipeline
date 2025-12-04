# src/final_training.py

# ============================
# ENTRENAMIENTO Y EVALUACIÓN FINAL
# ============================
# Fase 5: Entrena los modelos con los mejores hiperparámetros encontrados
# en la Fase 4 (GridSearch) y genera métricas finales completas.
#
# Para cada combinación (dataset, método_balanceo, tipo_escalado):
# - Carga los mejores hiperparámetros
# - Entrena el modelo final con TODO el conjunto de entrenamiento
# - Evalúa exhaustivamente en el conjunto de test
# - Genera reportes detallados y matrices de confusión
# ============================

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config import (
    PHASE_03_SCALING_DIR, 
    PHASE_04_HYPERPARAMETER_DIR,
    PHASE_05_FINAL_DIR
)


def load_scaled_data(dataset_name, method, scaler_type):
    """Carga los datos escalados"""
    base_path = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def load_best_hyperparameters(hp_dir, model_name):
    """
    Carga los mejores hiperparámetros encontrados en la Fase 4
    
    Args:
        hp_dir: Directorio donde están los resultados del GridSearch
        model_name: Nombre del modelo (svm, naive_bayes_gaussian, decision_tree)
    
    Returns:
        dict: Mejores hiperparámetros, o None si no existen
    """
    params_file = hp_dir / f"{model_name}_best_params.json"
    
    if not params_file.exists():
        print(f"      ⚠ No se encontraron hiperparámetros para {model_name}")
        return None
    
    try:
        with open(params_file, 'r') as f:
            data = json.load(f)
        return data['best_params']
    except Exception as e:
        print(f"      ✗ Error cargando hiperparámetros: {e}")
        return None


def create_model_with_params(model_name, best_params):
    """
    Crea un modelo con los mejores hiperparámetros
    
    Args:
        model_name: Nombre del modelo
        best_params: Diccionario con los hiperparámetros
    
    Returns:
        Modelo de sklearn configurado
    """
    if model_name == 'svm':
        return SVC(
            **best_params,
            random_state=42,
            cache_size=2000,
            max_iter=10000,
            probability=False,  # ≪≪ IMPORTANTE
        )
    elif model_name == 'naive_bayes_gaussian':
        return GaussianNB(**best_params)
    elif model_name == 'decision_tree':
        return DecisionTreeClassifier(**best_params, random_state=42)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


def calculate_comprehensive_metrics(y_test, y_pred):
    """
    Calcula todas las métricas requeridas desde la matriz de confusión
    
    Matriz de confusión:
                    Predicción
                    0(No-Buggy)  1(Buggy)
    Real  0(No-Buggy)    TN          FP
          1(Buggy)       FN          TP
    
    Args:
        y_test: Etiquetas reales
        y_pred: Predicciones del modelo
    
    Returns:
        dict: Todas las métricas calculadas
    """
    # Obtener matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Total de instancias
    N = len(y_test)
    
    # P (positivos reales) y N (negativos reales)
    P = tp + fn  # Total de casos buggy reales
    N_neg = tn + fp  # Total de casos no-buggy reales
    
    # P' (positivos predichos)
    P_pred = tp + fp
    
    # Métricas básicas desde la matriz de confusión
    metrics = {
        # Componentes de la matriz de confusión
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'P': int(P),  # Positivos reales
        'N': int(N_neg),  # Negativos reales
        'Total': int(N),
        
        # Error y Exactitud
        'Error': float((fp + fn) / N),
        'Exactitud': float((tp + tn) / N),  # = 1 - Error
        
        # Tasas
        'TP_Rate': float(tp / P) if P > 0 else 0.0,  # = Recall = Sensibilidad
        'FP_Rate': float(fp / N_neg) if N_neg > 0 else 0.0,
        
        # Precision y Recall
        'Precision': float(tp / P_pred) if P_pred > 0 else 0.0,  # tp / P'
        'Recall': float(tp / P) if P > 0 else 0.0,  # tp / P = TP-Rate
        
        # Sensibilidad y Especificidad
        'Sensibilidad': float(tp / P) if P > 0 else 0.0,  # = TP-Rate = Recall
        'Especificidad': float(tn / N_neg) if N_neg > 0 else 0.0,  # = 1 - FP-Rate
        
        # F1-Score (para consistencia)
        'F1_Score': float(f1_score(y_test, y_pred, zero_division=0)),
        
        # Accuracy (sklearn, para verificar)
        'Accuracy_sklearn': float(accuracy_score(y_test, y_pred)),
    }
    
    # Verificaciones
    assert abs(metrics['Exactitud'] - (1 - metrics['Error'])) < 1e-10, "Error en cálculo de Exactitud"
    assert abs(metrics['Exactitud'] - metrics['Accuracy_sklearn']) < 1e-10, "Error en cálculo de Accuracy"
    assert abs(metrics['TP_Rate'] - metrics['Recall']) < 1e-10, "Error en cálculo de TP-Rate"
    assert abs(metrics['TP_Rate'] - metrics['Sensibilidad']) < 1e-10, "Error en cálculo de Sensibilidad"
    assert abs(metrics['Especificidad'] - (1 - metrics['FP_Rate'])) < 1e-10, "Error en cálculo de Especificidad"
    
    return metrics, cm


def print_confusion_matrix_console(cm, model_name):
    """
    Imprime la matriz de confusión en consola de forma legible
    
    Args:
        cm: Matriz de confusión (2x2 numpy array)
        model_name: Nombre del modelo
    """
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n      {'='*50}")
    print(f"      MATRIZ DE CONFUSIÓN - {model_name.upper()}")
    print(f"      {'='*50}")
    print(f"                        Predicción")
    print(f"                   No-Buggy    Buggy")
    print(f"      Real  No-Buggy   {tn:6d}    {fp:6d}")
    print(f"            Buggy      {fn:6d}    {tp:6d}")
    print(f"      {'='*50}")
    print(f"      TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"      {'='*50}\n")


def print_metrics_console(metrics, model_name):
    """
    Imprime todas las métricas en consola de forma organizada
    
    Args:
        metrics: Diccionario con todas las métricas
        model_name: Nombre del modelo
    """
    print(f"\n      {'='*60}")
    print(f"      MÉTRICAS FINALES - {model_name.upper()}")
    print(f"      {'='*60}")
    
    print(f"\n      📊 Conteos:")
    print(f"         Total de instancias (N): {metrics['Total']}")
    print(f"         Positivos reales (P):    {metrics['P']} (Buggy)")
    print(f"         Negativos reales (N):    {metrics['N']} (No-Buggy)")
    
    print(f"\n      🎯 Matriz de Confusión:")
    print(f"         True Positives  (TP): {metrics['TP']}")
    print(f"         True Negatives  (TN): {metrics['TN']}")
    print(f"         False Positives (FP): {metrics['FP']}")
    print(f"         False Negatives (FN): {metrics['FN']}")
    
    print(f"\n      📈 Error y Exactitud:")
    print(f"         Error = (FP + FN) / N              = {metrics['Error']:.4f}")
    print(f"         Exactitud = (TP + TN) / N          = {metrics['Exactitud']:.4f}")
    print(f"         Verificación: 1 - Error            = {1 - metrics['Error']:.4f}")
    
    print(f"\n      🎲 Tasas:")
    print(f"         TP-Rate = TP / P                   = {metrics['TP_Rate']:.4f}")
    print(f"         FP-Rate = FP / N                   = {metrics['FP_Rate']:.4f}")
    
    print(f"\n      🔍 Precision y Recall:")
    print(f"         Precision = TP / P'                = {metrics['Precision']:.4f}")
    print(f"         Recall = TP / P (= TP-Rate)        = {metrics['Recall']:.4f}")
    
    print(f"\n      💡 Sensibilidad y Especificidad:")
    print(f"         Sensibilidad = TP / P (= TP-Rate)  = {metrics['Sensibilidad']:.4f}")
    print(f"         Especificidad = TN / N             = {metrics['Especificidad']:.4f}")
    print(f"         Verificación: 1 - FP-Rate          = {1 - metrics['FP_Rate']:.4f}")
    
    print(f"\n      ⭐ Otras Métricas:")
    print(f"         F1-Score                           = {metrics['F1_Score']:.4f}")
    
    print(f"\n      {'='*60}\n")


def evaluate_model_comprehensive(model, X_test, y_test, model_name):
    """
    Evaluación exhaustiva del modelo con todas las métricas requeridas
    
    Returns:
        tuple: (metrics, y_pred, y_proba, cm)
    """
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas desde matriz de confusión
    metrics, cm = calculate_comprehensive_metrics(y_test, y_pred)
    
    # Probabilidades (si el modelo lo soporta)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    
    # Métricas adicionales si tenemos probabilidades
    if y_proba is not None:
        try:
            metrics['ROC_AUC'] = float(roc_auc_score(y_test, y_proba))
            
            # Curvas ROC y Precision-Recall
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist()
            }
            metrics['PR_AUC'] = float(auc(recall_curve, precision_curve))
        except Exception as e:
            print(f"        ⚠ No se pudieron calcular ROC/PR curves: {e}")
    
    # Classification report detallado
    metrics['classification_report'] = classification_report(
        y_test, y_pred, 
        target_names=['No-Buggy', 'Buggy'],
        output_dict=True,
        zero_division=0
    )
    
    # Guardar matriz de confusión en formato lista
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics, y_pred, y_proba, cm


def plot_confusion_matrix(cm, output_path, title):
    """Genera y guarda la matriz de confusión"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No-Buggy', 'Buggy'],
        yticklabels=['No-Buggy', 'Buggy'],
        cbar_kws={'label': 'Cantidad'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    
    # Agregar totales
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.15, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}', 
             ha='center', transform=plt.gca().transAxes, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(metrics, output_dir, model_name):
    """Genera curvas ROC y Precision-Recall"""
    if 'roc_curve' not in metrics or 'pr_curve' not in metrics:
        return
    
    # Figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    roc_auc = metrics['ROC_AUC']
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FP-Rate)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TP-Rate)', fontsize=11)
    ax1.set_title(f'ROC Curve - {model_name}', fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision = metrics['pr_curve']['precision']
    recall = metrics['pr_curve']['recall']
    pr_auc = metrics['PR_AUC']
    
    ax2.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (TP-Rate)', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title(f'Precision-Recall Curve - {model_name}', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_roc_pr_curves.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions(y_test, y_pred, y_proba, output_dir, model_name):
    """Guarda las predicciones para análisis posterior"""
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
    })
    
    if y_proba is not None:
        predictions_df['y_proba'] = y_proba
    
    predictions_file = output_dir / f"{model_name}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)


def train_and_evaluate_model(dataset_name, method, scaler_type, model_name):
    """
    Entrena y evalúa un modelo específico con sus mejores hiperparámetros
    
    Args:
        dataset_name: Nombre del dataset
        method: Método de balanceo
        scaler_type: Tipo de escalado
        model_name: Nombre del modelo
    
    Returns:
        dict: Resultados completos de la evaluación
    """
    print(f"    → {model_name}: cargando datos y hiperparámetros...")
    
    # Cargar datos
    try:
        X_train, X_test, y_train, y_test = load_scaled_data(
            dataset_name, method, scaler_type
        )
    except FileNotFoundError:
        print(f"      ✗ Datos no encontrados")
        return None
    
    # Cargar mejores hiperparámetros
    hp_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    best_params = load_best_hyperparameters(hp_dir, model_name)
    
    if best_params is None:
        print(f"      ✗ No se pudo continuar sin hiperparámetros")
        return None
    
    print(f"      Hiperparámetros: {best_params}")
    
    # Crear modelo con mejores hiperparámetros
    try:
        model = create_model_with_params(model_name, best_params)
    except Exception as e:
        print(f"      ✗ Error creando modelo: {e}")
        return None
    
    # Entrenar con TODO el conjunto de entrenamiento
    print(f"      Entrenando modelo final...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluar exhaustivamente
    print(f"      Evaluando en test set...")
    metrics, y_pred, y_proba, cm = evaluate_model_comprehensive(
        model, X_test, y_test, model_name
    )
    
    # Mostrar matriz de confusión en consola
    print_confusion_matrix_console(cm, model_name)
    
    # Mostrar todas las métricas en consola
    print_metrics_console(metrics, model_name)
    
    # Preparar directorio de salida
    output_dir = PHASE_05_FINAL_DIR / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo entrenado
    model_file = output_dir / f"{model_name}_final_model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar métricas
    metrics['training_time_seconds'] = training_time
    metrics['best_hyperparameters'] = best_params
    metrics['timestamp'] = datetime.now().isoformat()
    
    metrics_file = output_dir / f"{model_name}_final_metrics.json"
    with open(metrics_file, 'w') as f:
        # Separar métricas serializables
        serializable_metrics = {
            k: v for k, v in metrics.items() 
            if k not in ['roc_curve', 'pr_curve']  # Muy grandes para JSON
        }
        json.dump(serializable_metrics, f, indent=2)
    
    # Generar visualizaciones
    plot_confusion_matrix(
        cm, 
        output_dir / f"{model_name}_confusion_matrix.png",
        f"Confusion Matrix - {model_name}\n{dataset_name} | {method} | {scaler_type}"
    )
    
    plot_roc_pr_curves(metrics, output_dir, model_name)
    
    # Guardar predicciones
    save_predictions(y_test, y_pred, y_proba, output_dir, model_name)
    
    print(f"    ✓ {model_name}: Entrenamiento y evaluación completados")
    print(f"       Tiempo de entrenamiento: {training_time:.2f}s\n")
    
    return metrics


def train_all_models_for_config(dataset_name, method, scaler_type):
    """
    Entrena todos los modelos (SVM, Naive Bayes, Decision Tree) 
    para una configuración específica
    """
    print(f"\n→ Entrenamiento final: {dataset_name} | {method} | {scaler_type}")
    
    models = ['svm', 'naive_bayes_gaussian', 'decision_tree']
    results = {}
    
    for model_name in models:
        result = train_and_evaluate_model(
            dataset_name, method, scaler_type, model_name
        )
        results[model_name] = result
    
    # Guardar resumen comparativo
    output_dir = PHASE_05_FINAL_DIR / method / dataset_name / scaler_type
    summary = {
        'dataset': dataset_name,
        'balancing_method': method,
        'scaler_type': scaler_type,
        'models': {
            model: {
                'F1_Score': res['F1_Score'],
                'Exactitud': res['Exactitud'],
                'Precision': res['Precision'],
                'Recall': res['Recall'],
                'Sensibilidad': res['Sensibilidad'],
                'Especificidad': res['Especificidad'],
                'Error': res['Error'],
                'TP_Rate': res['TP_Rate'],
                'FP_Rate': res['FP_Rate'],
                'ROC_AUC': res.get('ROC_AUC', None),
                'confusion_matrix': res['confusion_matrix']
            } if res else None
            for model, res in results.items()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


def train_final_models(dataset_name):
    """
    Entrena modelos finales para todas las combinaciones de un dataset
    (todos los métodos de balanceo y tipos de escalado)
    
    Args:
        dataset_name: Nombre del dataset a procesar
    """
    from src.config import BALANCING_METHODS, SCALING_TYPES
    
    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO FINAL DE MODELOS: {dataset_name}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            combination_key = f"{method}_{scaler_type}"
            print(f"\n{'-'*60}")
            print(f"Configuración: {combination_key}")
            print(f"{'-'*60}")
            
            results = train_all_models_for_config(dataset_name, method, scaler_type)
            all_results[combination_key] = results
    
    return all_results


if __name__ == "__main__":
    from src.config import DATASETS
    
    # Procesar el primer dataset como ejemplo
    if DATASETS:
        train_final_models(DATASETS[0])
    else:
        print("No hay datasets definidos en config.py")