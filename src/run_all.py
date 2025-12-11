# src/run_all.py

# ============================
# ORQUESTADOR DEL PIPELINE COMPLETO
# ============================
# Pipeline de 5 fases:
#
# Para cada dataset:
#   1. Preprocesamiento
#   2. Balanceo:
#       - unbalanced (sin resampling)
#       - CSBBoost
#       - HCBOU
#       - SMOTE
#   3. Escalado (StandardScaler, RobustScaler)
#   4. Búsqueda de hiperparámetros (GridSearchCV)
#   5. Entrenamiento final y evaluación
# ============================

from src.config import DATASETS
from src.preprocessing import preprocess_dataset
from src.balancing import (
    balance_with_csbboost,
    balance_with_hcbou,
    balance_with_smote,
    balance_without_resampling,  # baseline sin balanceo
)
from src.scaling import scale_balanced
from src.hyperparameter_tuning import tune_hyperparameters
from src.final_training import train_final_models
from src.logger import setup_logging, close_logging
from src.model_selection import select_best_models
from src.utils.label_checks import assert_buggy_label_is_one

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

            balance_without_resampling(ds)
            balance_with_csbboost(ds)
            balance_with_hcbou(ds)
            balance_with_smote(ds)

            # ----------------------------
            # FASE 3: ESCALADO
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 3: ESCALADO DE CARACTERÍSTICAS")
            print(f"{'='*80}")
            
            # Escalado para baseline sin balanceo
            scale_balanced(ds, method="unbalanced", scaler_type="standard")
            scale_balanced(ds, method="unbalanced", scaler_type="robust")

            # Escalado para CSBBoost
            scale_balanced(ds, method="csbboost", scaler_type="standard")
            scale_balanced(ds, method="csbboost", scaler_type="robust")

            # Escalado para HCBOU
            scale_balanced(ds, method="hcbou", scaler_type="standard")
            scale_balanced(ds, method="hcbou", scaler_type="robust")

            # Escalado para SMOTE
            scale_balanced(ds, method="smote", scaler_type="standard")
            scale_balanced(ds, method="smote", scaler_type="robust")

            # ----------------------------
            # FASE 4: BÚSQUEDA DE HIPERPARÁMETROS
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"FASE 4: BÚSQUEDA DE HIPERPARÁMETROS (GRIDSEARCH)")
            print(f"{'='*80}")
            assert_buggy_label_is_one(ds)
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

            # ----------------------------
            # FASE 6: Selección del mejor modelo final
            # ----------------------------
            
            print(f"\n{'='*80}")
            print(f"FASE 6: SELECCIÓN DEL MEJOR MODELO FINAL")
            print(f"{'='*80}")
            
            select_best_models(ds)
            
            # Si solo se desea ejecutar este paso para todos los datasets,
            # en terminal se hace de esta manera
            # python -m src.model_selection \ activemq-5.0.0 \ derby-10.5.1.1 \ groovy-1_6_BETA_1 \ hbase-0.94.0 \ hive-0.9.0 \ jruby-1.1 \ wicket-1.3.0-beta2

            
            # ----------------------------
            # FIN DEL PIPELINE COMPLETO
            # ----------------------------
            print(f"\n{'='*80}")
            print(f"✓ PIPELINE COMPLETO PARA {ds}")
            print(f"{'='*80}\n")

    finally:
        # Cerrar logging y restaurar stdout
        close_logging(tee_output)


if __name__ == "__main__":
    main()