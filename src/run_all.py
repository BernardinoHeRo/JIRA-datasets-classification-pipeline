# src/run_all.py

from src.config import DATASETS, BALANCING_METHODS
from src.preprocessing import preprocess_dataset
from src.balancing import (
    balance_with_csbboost,
    balance_with_hcbou,
    balance_with_smote,
    balance_without_resampling,
)
from src.scaling import scale_balanced, scale_fs_selected
from src.feature_selection import select_features_for_balanced
from src.logger import setup_logging, close_logging
from src.hyperparameter_tuning import (
    tune_hyperparameters,
    tune_hyperparameters_fs,
)
from src.final_training import train_final_models
from src.aggregate_results import main as aggregate_all_results

# ============================================================
# BANDERAS PARA ELEGIR QUÉ RUTA CORRER
# ============================================================
RUN_BASELINE_PIPELINE = False
RUN_FS_PIPELINE = True


def main():
    
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

            # --------------------------------------------------
            # FASE 1: PREPROCESAMIENTO (COMÚN)
            # --------------------------------------------------
            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print("FASE 1: PREPROCESAMIENTO")
            print(f"{'='*80}")
            preprocess_dataset(ds)

            # --------------------------------------------------
            # FASE 2: BALANCEO DE CLASES (COMÚN)
            # --------------------------------------------------
            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print("FASE 2: BALANCEO DE CLASES")
            print(f"{'='*80}")

            balance_without_resampling(ds)  # unbalanced
            balance_with_csbboost(ds)       # CSBBoost
            balance_with_hcbou(ds)          # HCBOU
            balance_with_smote(ds)          # SMOTE

            # ==================================================
            # RUTA A: PIPELINE BASELINE (SIN SELECCIÓN DE FEATURES)
            # ==================================================
            if RUN_BASELINE_PIPELINE:
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("RUTA A: PIPELINE SIN SELECCIÓN DE CARACTERÍSTICAS")
                print(f"{'='*80}")

                # ----------------------------
                # FASE 3: ESCALADO (SIN FS)
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 3: ESCALADO DE CARACTERÍSTICAS (SIN FS)")
                print(f"{'='*80}")

                for method in BALANCING_METHODS:
                    print(f"\n--- Escalado (SIN FS) para método: {method} ---")
                    scale_balanced(ds, method=method, scaler_type="standard")
                    scale_balanced(ds, method=method, scaler_type="robust")

                # ----------------------------
                # FASE 4: BÚSQUEDA DE HIPERPARÁMETROS (SIN FS)
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 4: BÚSQUEDA DE HIPERPARÁMETROS (GRIDSEARCH) - SIN FS")
                print(f"{'='*80}")

                tune_hyperparameters(ds, use_fs=False)

                # ----------------------------
                # FASE 5: ENTRENAMIENTO FINAL (SIN FS)
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 5: ENTRENAMIENTO FINAL CON MEJORES HIPERPARÁMETROS - SIN FS")
                print(f"{'='*80}")

                train_final_models(ds, use_fs=False)

                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"✓ RUTA A (SIN FS) COMPLETA PARA {ds}")
                print(f"{'='*80}\n")

            # ==================================================
            # RUTA B: PIPELINE CON SELECCIÓN DE CARACTERÍSTICAS
            # ==================================================
            if RUN_FS_PIPELINE:
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("RUTA B: PIPELINE CON SELECCIÓN DE CARACTERÍSTICAS")
                print(f"{'='*80}")

                # ----------------------------
                # FASE 3A (FS): SELECCIÓN DE CARACTERÍSTICAS
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 3A (FS): SELECCIÓN DE CARACTERÍSTICAS")
                print(f"{'='*80}")

                for method in BALANCING_METHODS:
                    print(f"\n--- [FS] Selección de features para método: {method} ---")
                    select_features_for_balanced(dataset_name=ds, method=method)

                # ----------------------------
                # FASE 3B (FS): ESCALADO CON FS
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 3B (FS): ESCALADO DE CONJUNTOS CON SELECCIÓN DE CARACTERÍSTICAS")
                print(f"{'='*80}")

                for method in BALANCING_METHODS:
                    print(f"\n--- [FS] Escalando (con FS) para método: {method} ---")
                    scale_fs_selected(ds, method=method, scaler_type="standard")
                    scale_fs_selected(ds, method=method, scaler_type="robust")

                # ----------------------------
                # FASE 4 (FS): BÚSQUEDA DE HIPERPARÁMETROS CON FS
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 4 (FS): BÚSQUEDA DE HIPERPARÁMETROS (GRIDSEARCH) CON FS")
                print(f"{'='*80}")

                tune_hyperparameters_fs(ds)

                # ----------------------------
                # FASE 5 (FS): ENTRENAMIENTO FINAL CON FS
                # ----------------------------
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print("FASE 5 (FS): ENTRENAMIENTO FINAL CON MEJORES HIPERPARÁMETROS (CON FS)")
                print(f"{'='*80}")

                train_final_models(ds, use_fs=True)

                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"\n{'I'*100}")
                print(f"✓ RUTA B (CON FS) COMPLETA PARA {ds}")
                print(f"{'='*80}\n")

            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print(f"\n{'I'*100}")
            print(f"✓ PIPELINE(S) TERMINADO(S) PARA {ds}")
            print(f"{'='*80}\n")

        # ==================================================
        # AL FINAL: AGREGAR RESULTADOS DE TODOS LOS DATASETS
        # ==================================================
        print("\n==================================================")
        print("=== GENERANDO ARCHIVO AGREGADO DE RESULTADOS ===")
        print("==================================================")
        aggregate_all_results()

    finally:
        close_logging(tee_output)


if __name__ == "__main__":
    main()