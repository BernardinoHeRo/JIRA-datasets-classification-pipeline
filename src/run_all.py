# src/run_all.py

# ============================
# ORQUESTADOR DEL PIPELINE
# ============================
# Aquí solo se define el flujo alto nivel:
#
# Para cada dataset:
#   1. Preprocesamiento
#   2. Balanceo (CSBBoost, HCBOU)
#   3. Escalado (StandardScaler, RobustScaler)
#   (en el futuro)
#   4. Selección de características
#   5. Búsqueda de hiperparámetros
#   6. Clasificación / métricas
# ============================

from src.config import DATASETS
from src.preprocessing import preprocess_dataset
from src.balancing import balance_with_csbboost, balance_with_hcbou, balance_with_smote
from src.scaling import scale_balanced
from src.hyperparameter_tuning import tune_hyperparameters
from src.logger import setup_logging, close_logging


def main():
    """
    Ejecuta el pipeline completo (fases 1 a 3) para cada dataset
    definido en src/config.py (lista DATASETS).
    """

    # Configurar logging al inicio
    tee_output = setup_logging()

    try:
        for ds in DATASETS:

            print("\n\n")
            print("*" * 150)
            print("*" * 150)
            print("*" * 150)
            print("*" * 150)
            print("*" * 150)
            print(f"\n================ DATASET: {ds} ================\n")

            # ----------------------------
            # 1) PREPROCESAMIENTO
            # ----------------------------
            preprocess_dataset(ds)

            # ----------------------------
            # 2) BALANCEO
            # ----------------------------
            balance_with_csbboost(ds)
            balance_with_hcbou(ds)
            balance_with_smote(ds)

            # ----------------------------
            # 3) ESCALADO
            # ----------------------------
            print(f"\n\n=====Escalado de datos con StandardScaler y RobustScaler=====")
            print("=============================================================")
            scale_balanced(ds, method="csbboost", scaler_type="standard")
            scale_balanced(ds, method="csbboost", scaler_type="robust")

            scale_balanced(ds, method="hcbou", scaler_type="standard")
            scale_balanced(ds, method="hcbou", scaler_type="robust")

            scale_balanced(ds, method="smote", scaler_type="standard")
            scale_balanced(ds, method="smote", scaler_type="robust")

            # ----------------------------
            # 4) FUTURO: SELECCIÓN DE CARACTERÍSTICAS
            # ----------------------------
            # print("[4] Selección de características")
            # run_feature_selection(ds, method="csbboost")
            # run_feature_selection(ds, method="hcbou")

            # ----------------------------
            # BÚSQUEDA DE HIPERPARÁMETROS
            # ----------------------------
            print("\n\n================ Búsqueda de hiperparámetros ================")
            print("=============================================================")

            # Ejecutar búsqueda de hiperparámetros para todas las combinaciones
            # de métodos de balanceo y tipos de escalado para este dataset
            tune_hyperparameters(ds)

            # ----------------------------
            # 5) FUTURO: CLASIFICACIÓN FINAL
            # ----------------------------
            # print("[5] Clasificación final y métricas")

    finally:
        # Cerrar logging y restaurar stdout
        close_logging(tee_output)


if __name__ == "__main__":
    main()