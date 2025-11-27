# JIRA Datasets Classification Pipeline

## 🌟 Highlights

- **Multi-phase ML pipeline** for software defect prediction using JIRA datasets
- **Advanced balancing techniques** including CSBBoost, HCBOU, and SMOTE
- **Automated feature scaling** with StandardScaler and RobustScaler
- **Comprehensive logging** with timestamped execution traces
- **Modular architecture** supporting easy extension to new phases
- **Research-ready** artifacts generated at each processing stage

## ℹ️ Overview

This project implements a comprehensive machine learning pipeline specifically designed for software defect prediction research using JIRA datasets. The pipeline processes software metrics from 7 different JIRA projects (ActiveMQ, Derby, Groovy, HBase, Hive, JRuby, and Wicket) to predict bugs using various data balancing and scaling techniques.

The pipeline is organized into sequential phases that transform raw CSV datasets into research-ready features through preprocessing, class balancing, and feature scaling. Each phase generates artifacts that can be used independently or fed into subsequent analysis phases. The system was developed for academic research in software engineering and provides a solid foundation for defect prediction experiments.

Unlike traditional ML pipelines, this system focuses specifically on the challenges of software defect datasets, including class imbalance and the need for robust feature scaling across different software metrics.

## 🚀 Usage

### Quick Start

Run the complete pipeline for all datasets:

```bash
python -m src.run_all
```

Run CSBBoost-specific pipeline:

```bash
python -m src.run_csbboost_pipeline
```

### Example Output Structure

After running the pipeline, you'll get organized artifacts:

```
artifacts/
├── 01_preprocessing/groovy-1_6_BETA_1/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── 02_balancing/hcbou/groovy-1_6_BETA_1/
│   ├── X_train_bal.csv
│   └── y_train_bal.csv
└── 03_scaling/hcbou/groovy-1_6_BETA_1/standard/
    ├── X_train_scaled.csv
    └── X_test_scaled.csv
```

### Pipeline Configuration

All settings are centralized in `src/config.py`:

```python
# Dataset configuration
DATASETS = ['activemq', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'wicket']
TARGET_COLUMN = 'RealBug'
EXCLUDED_COLUMNS = ['HeuBug', 'HeuBugCount', 'RealBugCount']

# Processing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

### Logging

All pipeline execution is automatically logged:

```bash
# View latest execution log
cat logs/log_YYYYMMDD_HHMMSS.log

# Monitor real-time execution
tail -f logs/log_YYYYMMDD_HHMMSS.log
```

## ⬇️ Installation

### Requirements

- **Python 3.12+**
- Required packages:
  ```bash
  pip install pandas numpy scikit-learn
  ```

### System Requirements

- **Memory**: Minimum 4GB RAM (8GB recommended for larger datasets)
- **Storage**: ~1GB for artifacts and logs
- **OS**: Cross-platform (Windows, macOS, Linux)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/BernardinoHeRo/JIRA-datasets-classification-pipeline.git
   cd JIRA-datasets-classification-pipeline
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn
   ```

3. Run the pipeline:
   ```bash
   python -m src.run_all
   ```

## 📊 Project Architecture

### Data Flow

```
datasets/ (Raw CSV files)
    ↓ preprocessing
artifacts/01_preprocessing/<dataset>/ (Train/test splits)
    ↓ balancing (CSBBoost, HCBOU, SMOTE)
artifacts/02_balancing/<method>/<dataset>/ (Balanced training data)
    ↓ scaling (Standard, Robust)
artifacts/03_scaling/<method>/<scaler>/<dataset>/ (Scaled features)
```

### Key Components

- **`src/run_all.py`** - Main pipeline orchestrator
- **`src/config.py`** - Centralized configuration
- **`src/logger.py`** - Logging utilities
- **`src/preprocessing.py`** - Data preprocessing and train/test splitting
- **`src/balancing.py`** - Class balancing techniques dispatcher
- **`src/scaling.py`** - Feature scaling with multiple scalers
- **Balancing implementations**:
  - `src/csbboost_impl.py` - CSBBoost with clustering
  - `src/hcbou_impl.py` - Hybrid Clustering-Based Oversampling
  - `src/smote_impl.py` - SMOTE implementation

## 🔬 Research Applications

This pipeline is designed for:

- **Software defect prediction** research
- **Class imbalance** technique comparison
- **Feature scaling** impact analysis
- **Cross-project defect prediction** studies
- **Ensemble method** development

### Supported Datasets

The pipeline works with JIRA datasets containing software metrics:

- ActiveMQ, Derby, Groovy, HBase, Hive, JRuby, Wicket
- Binary classification target: `RealBug` (0=no bug, 1=bug)
- Excludes heuristic columns to focus on metrics-based prediction

## 💭 Contribution & Feedback

We welcome contributions and feedback! Here's how you can help:

- **🐛 Found a bug?** [Open an issue](../../issues) with details and steps to reproduce
- **💡 Have an idea?** [Start a discussion](../../discussions) to share your thoughts
- **📈 Want to contribute?** Check our contribution guidelines and submit a pull request
- **📊 Research results?** We'd love to hear about your findings using this pipeline

### Future Extensions

The architecture supports easy addition of:

- Feature selection (phase 4)
- Hyperparameter tuning (phase 5)
- Classification and evaluation (phase 6)

## 📖 Further Reading

### Academic Context

This pipeline supports research in:

- Software engineering empirical studies
- Defect prediction methodologies
- Class imbalance handling in SE datasets

### Related Work

- [Software Defect Prediction: A Survey](https://example.com)
- [Class Imbalance in Software Engineering](https://example.com)
- [JIRA Mining for Software Analytics](https://example.com)

### Documentation

- Check `logs/` for execution traces and debugging information
- Review `src/config.py` for all configurable parameters

---

**Made with ❤️ for software engineering research**

_This pipeline was developed as part of academic research in software defect prediction. If you use this code in your research, please consider citing our work._
