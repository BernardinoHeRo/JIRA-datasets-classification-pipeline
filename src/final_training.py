# src/final_training.py
# =============================================================================
# FASE 5: ENTRENAMIENTO Y EVALUACIÓN FINAL
# =============================================================================
# Objetivo:
#   - Usar los mejores hiperparámetros encontrados en Fase 4 (GridSearch).
#   - Entrenar modelos finales (SVM, Naive Bayes, Decision Tree) con TODO el
#     conjunto de entrenamiento escalado.
#   - Evaluar exhaustivamente en el conjunto de prueba.
#   - Generar:
#       * Matriz de confusión (filas = reales, columnas = predichas)
#       * Métricas derivadas de la matriz de confusión:
#           - Exactitud / Accuracy
#           - Precision, Recall, Sensibilidad, Especificidad
#           - F1-Score
#           - F2-Score (β = 2, p/ clase Buggy)
#           - Balanced Accuracy, BER (Balanced Error Rate)
#           - MCC (Matthews Correlation Coefficient)
#       * Curvas ROC / PR (si aplica)
#       * Archivos: métricas JSON, predicciones CSV, figuras PNG, modelo .pkl
# =============================================================================

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    fbeta_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    PHASE_03_SCALING_DIR,
    PHASE_04_HYPERPARAMETER_DIR,
    PHASE_05_FINAL_DIR,
)

BUGGY_LABEL = 1      # misma convención: 1 = Buggy
F_BETA = 2.0         # mismo beta que en hyperparameter_tuning

# =============================================================================
# 1. CARGA DE DATOS Y MEJORES HIPERPARÁMETROS
# =============================================================================


def load_scaled_data(dataset_name: str, method: str, scaler_type: str):
    """
    Carga los datos escalados para una configuración dada.

    Estructura esperada:
        artifacts/03_scaling/<method>/<dataset_name>/<scaler_type>/
            - X_train_scaled.csv
            - X_test_scaled.csv
            - y_train.csv
            - y_test.csv

    Convención de etiquetas:
        - 0 = No-Buggy
        - 1 = Buggy
    """
    base_path = PHASE_03_SCALING_DIR / method / dataset_name / scaler_type

    X_train = pd.read_csv(base_path / "X_train_scaled.csv")
    X_test = pd.read_csv(base_path / "X_test_scaled.csv")
    y_train = pd.read_csv(base_path / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(base_path / "y_test.csv").squeeze("columns")

    # 🆕 Aseguramos que las etiquetas sean enteros {0,1}
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    unique_train = sorted(y_train.unique().tolist())
    unique_test = sorted(y_test.unique().tolist())
    print(f"      📌 [Fase 5] Etiquetas únicas y_train: {unique_train}")
    print(f"      📌 [Fase 5] Etiquetas únicas y_test : {unique_test}")

    if not set(unique_train).issubset({0, 1}) or not set(unique_test).issubset({0, 1}):
        raise ValueError(
            f"      ❌ [Fase 5] Las etiquetas no están en {{0,1}}.\n"
            f"         y_train: {unique_train} | y_test: {unique_test}\n"
            f"         Verifica que RealBug se mantenga como 0/1 en el pipeline."
        )

    return X_train, X_test, y_train, y_test


def load_best_hyperparameters(hp_dir: Path, model_name: str):
    """
    Carga los mejores hiperparámetros encontrados en la Fase 4.

    Args:
        hp_dir: Directorio de resultados del GridSearch para la configuración:
                artifacts/04_hyperparameter_tuning/<method>/<dataset>/<scaler>/
        model_name: 'svm', 'naive_bayes_gaussian' o 'decision_tree'

    Returns:
        dict con los mejores hiperparámetros, o None si no existen.
    """
    params_file = hp_dir / f"{model_name}_best_params.json"

    if not params_file.exists():
        print(f"      ⚠ No se encontraron hiperparámetros para {model_name}")
        return None

    try:
        with open(params_file, "r") as f:
            data = json.load(f)
        return data.get("best_params", None)
    except Exception as e:
        print(f"      ✗ Error cargando hiperparámetros para {model_name}: {e}")
        return None


# =============================================================================
# 2. CREACIÓN DE MODELOS CON HIPERPARÁMETROS ÓPTIMOS
# =============================================================================


def create_model_with_params(model_name: str, best_params: dict):
    """
    Crea un modelo de sklearn con los mejores hiperparámetros.

    Nota:
        - SVM se configura sin probability=True para no penalizar el rendimiento.
          Si se necesita ROC/PR exacto, se puede activar probability=True,
          considerando el costo computacional.
    """
    if model_name == "svm":
        return SVC(
            **best_params,
            random_state=42,
            cache_size=2000,
            max_iter=10000,
            probability=False,
        )
    elif model_name == "naive_bayes_gaussian":
        return GaussianNB(**best_params)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**best_params, random_state=42)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")


# =============================================================================
# 3. CÁLCULO DE MÉTRICAS A PARTIR DE LA MATRIZ DE CONFUSIÓN
# =============================================================================


def calculate_comprehensive_metrics(y_test, y_pred):
    """
    Calcula todas las métricas a partir de la matriz de confusión binaria.

    Convención (binaria, etiquetas {0,1}):
        - 0 = No-Buggy
        - 1 = Buggy

    Matriz de confusión (sklearn):
                        Predicción
                        0(No-Buggy)  1(Buggy)
        Real 0(No-Buggy)    TN          FP
             1(Buggy)       FN          TP

    Returns:
        metrics (dict), cm (np.ndarray 2x2)
    """
    # Forzamos el orden de etiquetas para asegurar layout 2x2 coherente
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if cm.shape != (2, 2):
        raise ValueError(
            f"Se esperaba matriz de confusión 2x2 para problema binario, "
            f"pero se obtuvo shape={cm.shape}"
        )

    tn, fp, fn, tp = cm.ravel()

    # Total de instancias
    N = len(y_test)

    # P (positivos reales) y N_neg (negativos reales)
    P = tp + fn  # Total de casos buggy reales
    N_neg = tn + fp  # Total de casos no-buggy reales

    # P' (positivos predichos)
    P_pred = tp + fp

    # Métricas derivadas de la matriz de confusión
    metrics = {
        # Componentes básicos
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "P": int(P),           # Positivos reales (Buggy)
        "N": int(N_neg),       # Negativos reales (No-Buggy)
        "Total": int(N),
        # Error y exactitud
        "Error": float((fp + fn) / N) if N > 0 else 0.0,
        "Exactitud": float((tp + tn) / N) if N > 0 else 0.0,
        # Tasas básicas
        "TP_Rate": float(tp / P) if P > 0 else 0.0,       # = Recall = Sensibilidad
        "FP_Rate": float(fp / N_neg) if N_neg > 0 else 0.0,
        # Precisión y Recall
        "Precision": float(tp / P_pred) if P_pred > 0 else 0.0,
        "Recall": float(tp / P) if P > 0 else 0.0,
        # Sensibilidad / Especificidad
        "Sensibilidad": float(tp / P) if P > 0 else 0.0,
        "Especificidad": float(tn / N_neg) if N_neg > 0 else 0.0,
        # F1 y Accuracy de sklearn (para verificar)
        "F1_Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "Accuracy_sklearn": float(accuracy_score(y_test, y_pred)),
    }

    # F2-Score (β = 2) para la clase positiva (Buggy)
    metrics["F2_Score"] = float(
        fbeta_score(
            y_test,
            y_pred,
            beta=F_BETA,
            pos_label=BUGGY_LABEL,
            average="binary",
            zero_division=0,
        )
    )

    # Balanced Accuracy y BER (Balanced Error Rate)
    if P > 0 and N_neg > 0:
        balanced_accuracy = 0.5 * (metrics["TP_Rate"] + metrics["Especificidad"])
        ber = 1.0 - balanced_accuracy
    else:
        balanced_accuracy = 0.0
        ber = 0.0

    metrics["Balanced_Accuracy"] = float(balanced_accuracy)
    metrics["BER"] = float(ber)

    # MCC (Matthews Correlation Coefficient)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom > 0:
        mcc = ((tp * tn) - (fp * fn)) / denom
    else:
        mcc = 0.0
    metrics["MCC"] = float(mcc)

    # Alias explícito de Accuracy (además de Exactitud y Accuracy_sklearn)
    metrics["Accuracy"] = metrics["Accuracy_sklearn"]

    # Comprobaciones internas de consistencia
    assert abs(metrics["Exactitud"] - (1 - metrics["Error"])) < 1e-10, \
        "Inconsistencia: Exactitud != 1 - Error"
    assert abs(metrics["Exactitud"] - metrics["Accuracy_sklearn"]) < 1e-10, \
        "Inconsistencia: Exactitud != Accuracy_sklearn"
    assert abs(metrics["TP_Rate"] - metrics["Recall"]) < 1e-10, \
        "Inconsistencia: TP_Rate != Recall"
    assert abs(metrics["TP_Rate"] - metrics["Sensibilidad"]) < 1e-10, \
        "Inconsistencia: TP_Rate != Sensibilidad"
    assert abs(metrics["Especificidad"] - (1 - metrics["FP_Rate"])) < 1e-10, \
        "Inconsistencia: Especificidad != 1 - FP_Rate"

    return metrics, cm


def print_confusion_matrix_console(cm: np.ndarray, model_name: str):
    """
    Imprime la matriz de confusión en consola de forma legible.

    Layout:
                        Predicción
                        No-Buggy    Buggy
        Real  No-Buggy     TN         FP
              Buggy        FN         TP
    """
    tn, fp, fn, tp = cm.ravel()

    print(f"\n      {'=' * 52}")
    print(f"      MATRIZ DE CONFUSIÓN - {model_name.upper()}")
    print(f"      (filas = reales, columnas = predichas)")
    print(f"      {'=' * 52}")
    print(f"                          Predicción")
    print(f"                     No-Buggy    Buggy")
    print(f"      Real  No-Buggy   {tn:7d}   {fp:7d}")
    print(f"            Buggy      {fn:7d}   {tp:7d}")
    print(f"      {'=' * 52}")
    print(f"      TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"      {'=' * 52}\n")


def print_metrics_console(metrics: dict, model_name: str):
    """Imprime las métricas finales de forma organizada y compacta."""
    print(f"\n      {'=' * 60}")
    print(f"      MÉTRICAS FINALES - {model_name.upper()}")
    print(f"      {'=' * 60}")

    print(f"\n      📊 Conteos:")
    print(f"         Total de instancias (N): {metrics['Total']}")
    print(f"         Positivos reales (P):    {metrics['P']} (Buggy)")
    print(f"         Negativos reales (N):    {metrics['N']} (No-Buggy)")

    print(f"\n      🎯 Matriz de Confusión:")
    print(f"         TP: {metrics['TP']}   TN: {metrics['TN']}")
    print(f"         FP: {metrics['FP']}   FN: {metrics['FN']}")

    print(f"\n      📈 Error y Exactitud:")
    print(f"         Error         = (FP + FN) / N   = {metrics['Error']:.4f}")
    print(f"         Exactitud     = (TP + TN) / N   = {metrics['Exactitud']:.4f}")
    print(f"         Accuracy      (sklearn)         = {metrics['Accuracy']:.4f}")

    print(f"\n      🎲 Tasas:")
    print(f"         TP-Rate (Recall)                = {metrics['TP_Rate']:.4f}")
    print(f"         FP-Rate                         = {metrics['FP_Rate']:.4f}")

    print(f"\n      🔍 Precisión y Recall:")
    print(f"         Precision = TP / P'             = {metrics['Precision']:.4f}")
    print(f"         Recall    = TP / P              = {metrics['Recall']:.4f}")

    print(f"\n      💡 Sensibilidad y Especificidad:")
    print(f"         Sensibilidad (TP/P)             = {metrics['Sensibilidad']:.4f}")
    print(f"         Especificidad (TN/N)            = {metrics['Especificidad']:.4f}")

    print(f"\n      ⚖️ Balanced Accuracy y BER:")
    print(f"         Balanced Accuracy               = {metrics['Balanced_Accuracy']:.4f}")
    print(f"         BER (Balanced Error Rate)       = {metrics['BER']:.4f}")

    print(f"\n      🧮 MCC:")
    print(f"         MCC (Matthews Corr. Coef.)      = {metrics['MCC']:.4f}")

    print(f"\n      ⭐ F1-Score:")
    print(f"         F1_Score                        = {metrics['F1_Score']:.4f}")

    print(f"\n      ⭐⭐ F2-Score (β=2, Buggy):")
    print(f"         F2_Score                        = {metrics['F2_Score']:.4f}")

    print(f"\n      {'=' * 60}\n")


# =============================================================================
# 4. EVALUACIÓN EXHAUSTIVA (INCLUYE ROC/PR SI ES POSIBLE)
# =============================================================================


def evaluate_model_comprehensive(model, X_test, y_test, model_name: str):
    """
    Evalúa el modelo con todas las métricas requeridas.

    Returns:
        metrics (dict),
        y_pred (np.ndarray),
        y_proba (np.ndarray o None),
        cm (np.ndarray 2x2)
    """
    # Predicciones puntuales
    y_pred = model.predict(X_test)

    # Métricas basadas en la matriz de confusión
    metrics, cm = calculate_comprehensive_metrics(y_test, y_pred)

    # Probabilidades o scores (si el modelo lo soporta)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    # Métricas dependientes de probabilidad
    if y_proba is not None:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_test, y_proba))

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            metrics["pr_curve"] = {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
            }
            metrics["PR_AUC"] = float(auc(recall_curve, precision_curve))
        except Exception as e:
            print(f"        ⚠ No se pudieron calcular ROC/PR curves: {e}")

    # Classification report detallado (para análisis posterior)
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        target_names=["No-Buggy", "Buggy"],
        output_dict=True,
        zero_division=0,
    )

    # Guardar matriz de confusión como lista serializable
    metrics["confusion_matrix"] = cm.tolist()

    return metrics, y_pred, y_proba, cm


# =============================================================================
# 5. GRÁFICAS Y PREDICCIONES
# =============================================================================


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, title: str):
    """
    Genera y guarda la matriz de confusión (2x2) como figura PNG.

    Filas = reales, columnas = predichas.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No-Buggy", "Buggy"],
        yticklabels=["No-Buggy", "Buggy"],
        cbar_kws={"label": "Cantidad"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Clase Real", fontsize=12)
    plt.xlabel("Clase Predicha", fontsize=12)

    tn, fp, fn, tp = cm.ravel()
    plt.text(
        0.5,
        -0.15,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        ha="center",
        transform=plt.gca().transAxes,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(metrics: dict, output_dir: Path, model_name: str):
    """Genera curvas ROC y Precision-Recall si hay información suficiente."""
    if "roc_curve" not in metrics or "pr_curve" not in metrics:
        return

    fpr = metrics["roc_curve"]["fpr"]
    tpr = metrics["roc_curve"]["tpr"]
    roc_auc = metrics.get("ROC_AUC", None)

    precision = metrics["pr_curve"]["precision"]
    recall = metrics["pr_curve"]["recall"]
    pr_auc = metrics.get("PR_AUC", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax1.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC (AUC = {roc_auc:.3f})" if roc_auc is not None else "ROC",
    )
    ax1.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("False Positive Rate (FP-Rate)")
    ax1.set_ylabel("True Positive Rate (TP-Rate)")
    ax1.set_title(f"ROC Curve - {model_name}")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Precision-Recall
    ax2.plot(
        recall,
        precision,
        lw=2,
        label=f"PR (AUC = {pr_auc:.3f})" if pr_auc is not None else "PR",
    )
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlabel("Recall (TP-Rate)")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve - {model_name}")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_roc_pr_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_predictions(y_test, y_pred, y_proba, output_dir: Path, model_name: str):
    """Guarda predicciones y, si existe, probabilidad/score asociado."""
    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    if y_proba is not None:
        predictions_df["y_proba"] = y_proba

    predictions_file = output_dir / f"{model_name}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)


# =============================================================================
# 6. ENTRENAMIENTO Y EVALUACIÓN POR MODELO / CONFIGURACIÓN
# =============================================================================


def train_and_evaluate_model(
    dataset_name: str, method: str, scaler_type: str, model_name: str
):
    """
    Entrena y evalúa un modelo específico con sus mejores hiperparámetros.

    Flujo:
        1) Cargar datos escalados (Fase 3).
        2) Cargar mejores hiperparámetros (Fase 4).
        3) Entrenar modelo final con TODO el train.
        4) Evaluar en test y generar artefactos.
    """
    print(f"    → {model_name}: cargando datos y mejores hiperparámetros...")

    # 1) Cargar datos escalados
    try:
        X_train, X_test, y_train, y_test = load_scaled_data(
            dataset_name, method, scaler_type
        )
    except FileNotFoundError:
        print(
            f"      ✗ Datos no encontrados para {dataset_name} | {method} | {scaler_type}"
        )
        return None

    # 2) Cargar mejores hiperparámetros
    hp_dir = PHASE_04_HYPERPARAMETER_DIR / method / dataset_name / scaler_type
    best_params = load_best_hyperparameters(hp_dir, model_name)
    if best_params is None:
        print(f"      ✗ No se pudo continuar sin hiperparámetros para {model_name}")
        return None

    print(f"      Mejores hiperparámetros: {best_params}")

    # 3) Crear modelo y entrenar
    try:
        model = create_model_with_params(model_name, best_params)
    except Exception as e:
        print(f"      ✗ Error creando modelo {model_name}: {e}")
        return None

    print(f"      Entrenando modelo final...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()

    # 4) Evaluar en test
    print(f"      Evaluando en conjunto de prueba...")
    metrics, y_pred, y_proba, cm = evaluate_model_comprehensive(
        model, X_test, y_test, model_name
    )

    # Logs legibles en consola
    print_confusion_matrix_console(cm, model_name)
    print_metrics_console(metrics, model_name)

    # Directorio de salida para la Fase 5
    output_dir = PHASE_05_FINAL_DIR / method / dataset_name / scaler_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo final
    model_file = output_dir / f"{model_name}_final_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Guardar métricas (sin curvas para no inflar el JSON)
    metrics["training_time_seconds"] = training_time
    metrics["best_hyperparameters"] = best_params
    metrics["timestamp"] = datetime.now().isoformat()

    metrics_file = output_dir / f"{model_name}_final_metrics.json"
    serializable_metrics = {
        k: v for k, v in metrics.items() if k not in ["roc_curve", "pr_curve"]
    }
    with open(metrics_file, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Figuras y predicciones
    plot_confusion_matrix(
        cm,
        output_dir / f"{model_name}_confusion_matrix.png",
        f"Confusion Matrix - {model_name}\n{dataset_name} | {method} | {scaler_type}",
    )
    plot_roc_pr_curves(metrics, output_dir, model_name)
    save_predictions(y_test, y_pred, y_proba, output_dir, model_name)

    print(f"    ✓ {model_name}: entrenamiento y evaluación completados")
    print(f"      Tiempo de entrenamiento: {training_time:.2f}s\n")

    return metrics


def train_all_models_for_config(dataset_name: str, method: str, scaler_type: str):
    """
    Entrena todos los modelos (SVM, Naive Bayes, Decision Tree)
    para una configuración específica (dataset, método de balanceo, escalado).
    """
    print(f"\n→ Entrenamiento final: {dataset_name} | {method} | {scaler_type}")

    models = ["svm", "naive_bayes_gaussian", "decision_tree"]
    results = {}

    for model_name in models:
        res = train_and_evaluate_model(dataset_name, method, scaler_type, model_name)
        results[model_name] = res

    # Resumen comparativo por configuración
    output_dir = PHASE_05_FINAL_DIR / method / dataset_name / scaler_type
    summary = {
        "dataset": dataset_name,
        "balancing_method": method,
        "scaler_type": scaler_type,
        "models": {
            model: {
                "F1_Score": r["F1_Score"],
                "F2_Score": r["F2_Score"],
                "Exactitud": r["Exactitud"],
                "Accuracy": r["Accuracy"],
                "Balanced_Accuracy": r["Balanced_Accuracy"],
                "BER": r["BER"],
                "MCC": r["MCC"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "Sensibilidad": r["Sensibilidad"],
                "Especificidad": r["Especificidad"],
                "Error": r["Error"],
                "TP_Rate": r["TP_Rate"],
                "FP_Rate": r["FP_Rate"],
                "ROC_AUC": r.get("ROC_AUC", None),
                "confusion_matrix": r["confusion_matrix"],
            }
            if r
            else None
            for model, r in results.items()
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


# =============================================================================
# 7. ORQUESTADOR DE FASE 5 PARA UN DATASET
# =============================================================================


def train_final_models(dataset_name: str):
    """
    Ejecuta la Fase 5 para un dataset:
        - Recorre todos los métodos de balanceo y tipos de escalado
        - Entrena y evalúa todos los modelos para cada combinación
    """
    from src.config import BALANCING_METHODS, SCALING_TYPES

    print(f"\n{'=' * 80}")
    print(f"ENTRENAMIENTO FINAL DE MODELOS - DATASET: {dataset_name}")
    print(f"{'=' * 80}")

    all_results = {}

    for method in BALANCING_METHODS:
        for scaler_type in SCALING_TYPES:
            combination_key = f"{method}_{scaler_type}"
            print(f"\n{'-' * 60}")
            print(f"Configuración: {combination_key}")
            print(f"{'-' * 60}")

            cfg_results = train_all_models_for_config(
                dataset_name, method, scaler_type
            )
            all_results[combination_key] = cfg_results

    return all_results


if __name__ == "__main__":
    from src.config import DATASETS

    if DATASETS:
        train_final_models(DATASETS[0])
    else:
        print("No hay datasets definidos en config.py")