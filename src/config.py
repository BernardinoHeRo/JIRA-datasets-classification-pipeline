# src/config.py

# ============================
# CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================
# Aquí centralizamos:
# - La ruta base del proyecto
# - Carpetas de datasets, artifacts y plots
# - Nombre de la columna objetivo
# - Columnas a eliminar
# - Lista de nombres de datasets
# ============================

from pathlib import Path

# Ruta base del proyecto (la carpeta "pipeline_jira_cassification")
BASE_DIR = Path(__file__).resolve().parents[1]

# Carpeta donde están tus CSV originales
DATASETS_DIR = BASE_DIR / "datasets"

# Carpeta donde se guardarán todos los resultados intermedios
# organizados por fases (01_preprocessing, 02_balancing, etc.)
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Carpeta opcional para guardar figuras (si luego las usas)
PLOTS_DIR = BASE_DIR / "plots"

# Carpeta para guardar archivos de log
LOGS_DIR = BASE_DIR / "logs"

# Carpetas para las fases del pipeline organizadas por número
PHASE_01_PREPROCESSING_DIR = ARTIFACTS_DIR / "01_preprocessing"
PHASE_02_BALANCING_DIR = ARTIFACTS_DIR / "02_balancing"
PHASE_03_SCALING_DIR = ARTIFACTS_DIR / "03_scaling"
PHASE_04_HYPERPARAMETER_DIR = ARTIFACTS_DIR / "04_hyperparameter_tuning"

# Nombre de la columna objetivo en tus datasets JIRA
TARGET_COL = "RealBug"

# Columnas que no se usarán como predictores
COLUMNS_TO_DROP = ["HeuBug", "HeuBugCount", "RealBugCount"]

# Lista de datasets (nombres de archivo sin extensión .csv)
DATASETS = [
    "activemq-5.0.0",
    "derby-10.5.1.1",
    "groovy-1_6_BETA_1",
    "hbase-0.94.0",
    "hive-0.9.0",
    "jruby-1.1",
    "wicket-1.3.0-beta2",
]

# Métodos de balanceo disponibles
BALANCING_METHODS = ["csbboost", "hcbou", "smote"]

# Tipos de escalado disponibles
SCALING_TYPES = ["standard", "robust"]

# Modelos de clasificación para hyperparameter tuning
CLASSIFICATION_MODELS = ["svm", "naive_bayes", "decision_tree"]