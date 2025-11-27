# src/run_csbboost_pipeline.py
from config import DATASETS
from preprocessing_csbboost import run_csbboost_pipeline_for_dataset

def main():
    for ds_name in DATASETS:
        run_csbboost_pipeline_for_dataset(ds_name)

if __name__ == "__main__":
    main()