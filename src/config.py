# src/config.py

from pathlib import Path

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]

# Carpetas principales
DATASETS_DIR = BASE_DIR / "datasets"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# Carpetas por fase
PHASE_01_PREPROCESSING_DIR = ARTIFACTS_DIR / "01_preprocessing"
PHASE_02_BALANCING_DIR = ARTIFACTS_DIR / "02_balancing"
PHASE_03_FEATURE_SELECTION_DIR = ARTIFACTS_DIR / "03a_feature_selection"
PHASE_03_FS_SCALING_DIR = ARTIFACTS_DIR / "03b_scaling_fs"
PHASE_03_SCALING_DIR = ARTIFACTS_DIR / "03_scaling"

# Hiperparámetros (ruta A y ruta B)
PHASE_04_HYPERPARAMETER_DIR = ARTIFACTS_DIR / "04_hyperparameter_tuning"
PHASE_04_HYPERPARAMETER_FS_DIR = ARTIFACTS_DIR / "04_hyperparameter_tuning_fs"

PHASE_05_FINAL_DIR = ARTIFACTS_DIR / "05_final_models"

# Nombre de la columna objetivo
TARGET_COL = "RealBug"

# Columnas a eliminar
COLUMNS_TO_DROP = ["HeuBug", "HeuBugCount", "RealBugCount"]

# Datasets
DATASETS = [
    "activemq-5.0.0",
    "derby-10.5.1.1",
    "groovy-1_6_BETA_1",
    "hbase-0.94.0",
    "hive-0.9.0",
    "jruby-1.1",
    "wicket-1.3.0-beta2",
]

# Métodos de balanceo
BALANCING_METHODS = ["unbalanced", "csbboost", "hcbou", "smote"]

# Tipos de escalado
SCALING_TYPES = ["standard", "robust"]

# Modelos de clasificación (solo referencia)
CLASSIFICATION_MODELS = ["svm", "naive_bayes", "decision_tree"]