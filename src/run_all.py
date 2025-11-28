# src/run_all.py

# ============================
# ORQUESTADOR DEL PIPELINE COMPLETO
# ============================
# Pipeline de 5 fases:
#
# Para cada dataset:
#   1. Preprocesamiento
#   2. Balanceo (CSBBoost, HCBOU, SMOTE)
#   3. Escalado (StandardScaler, RobustScaler)
#   4. Búsqueda de hiperparámetros (GridSearchCV)
#   5. Entrenamiento final y evaluación
# ============================

from src.config import DATASETS
from src.preprocessing import preprocess_dataset
from src.balancing import balance_with_csbboost, balance_with_hcbou, balance_with_smote
from src.scaling import scale_balanced
from src.hyperparameter_tuning import tune_hyperparameters
from src.final_training import train_final_models
from src.logger import setup_logging, close_logging


def main():
    """
    Ejecuta el pipeline completo (fases 1 a 5) para cada dataset
    definido en src/config.py (lista DATASETS).
    """

    # Configurar logging al inicio
    tee_output = setup_logging()

    try:
        for ds in DATASETS:

            print("\n\n")
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print("*" * 200)
            print(f"\n================ DATASET: {ds} ================\n")

            # ----------------------------
            # FASE 1: PREPROCESAMIENTO
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 1: PREPROCESAMIENTO")
            print(f"{'='*80}")
            preprocess_dataset(ds)

            # ----------------------------
            # FASE 2: BALANCEO
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 2: BALANCEO DE CLASES")
            print(f"{'='*80}")
            balance_with_csbboost(ds)
            balance_with_hcbou(ds)
            balance_with_smote(ds)

            # ----------------------------
            # FASE 3: ESCALADO
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 3: ESCALADO DE CARACTERÍSTICAS")
            print(f"{'='*80}")
            
            scale_balanced(ds, method="csbboost", scaler_type="standard")
            scale_balanced(ds, method="csbboost", scaler_type="robust")

            scale_balanced(ds, method="hcbou", scaler_type="standard")
            scale_balanced(ds, method="hcbou", scaler_type="robust")

            scale_balanced(ds, method="smote", scaler_type="standard")
            scale_balanced(ds, method="smote", scaler_type="robust")

            # ----------------------------
            # FASE 4: BÚSQUEDA DE HIPERPARÁMETROS
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 4: BÚSQUEDA DE HIPERPARÁMETROS (GRIDSEARCH)")
            print(f"{'='*80}")

            # Ejecutar búsqueda de hiperparámetros para todas las combinaciones
            # (métodos de balanceo × tipos de escalado) para este dataset
            tune_hyperparameters(ds)

            # ----------------------------
            # FASE 5: ENTRENAMIENTO FINAL Y EVALUACIÓN
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 5: ENTRENAMIENTO FINAL CON MEJORES HIPERPARÁMETROS")
            print(f"{'='*80}")

            # Entrenar modelos finales con los mejores hiperparámetros encontrados
            train_final_models(ds)

            print(f"\n{'='*80}")
            print(f"✓ PIPELINE COMPLETO PARA {ds}")
            print(f"{'='*80}\n")

    finally:
        # Cerrar logging y restaurar stdout
        close_logging(tee_output)


if __name__ == "__main__":
    main()