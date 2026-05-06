.PHONY: check compile test env-check mlflow-ui eval-v3-mlflow train-v3-mlflow

PYTHON ?= python3
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
MLFLOW_EXPERIMENT ?= v3_evaluation
TRAIN_MLFLOW_EXPERIMENT ?= v3_training
CHECKPOINT ?= artifacts/models/v3_cough.pt
EVAL_ARGS ?=
TRAIN_ARGS ?=

check: compile env-check

compile:
	$(PYTHON) -m py_compile data/prepate_dataset.py data/save_raw_data.py experiments/data_utils.py report/generate_figures.py scripts/prepare_dataset.py scripts/check_environment.py scripts/train_v3.py scripts/evaluate_v3.py scripts/predict_record.py scripts/dataset_summary.py scripts/error_analysis_v3.py scripts/sweep_event_boundaries_v3.py src/cough_analysis/*.py tests/*.py

env-check:
	PYTHONPATH=src $(PYTHON) scripts/check_environment.py

test:
	PYTHONPATH=src pytest

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

eval-v3-mlflow:
	PYTHONPATH=src $(PYTHON) scripts/evaluate_v3.py --checkpoint $(CHECKPOINT) --mlflow --mlflow-experiment $(MLFLOW_EXPERIMENT) --mlflow-tracking-uri $(MLFLOW_TRACKING_URI) $(EVAL_ARGS)

train-v3-mlflow:
	PYTHONPATH=src $(PYTHON) scripts/train_v3.py --mlflow --mlflow-experiment $(TRAIN_MLFLOW_EXPERIMENT) --mlflow-tracking-uri $(MLFLOW_TRACKING_URI) $(TRAIN_ARGS)
