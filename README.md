# Multi-Sensor Cough Analysis and State Classification

This repository contains the EE491-EE492 senior design project on cough detection and activity classification using synchronized wearable sensor recordings.

## Project Layout

```text
data/           Dataset metadata, raw CSV folders, and curated CSV records
docs/           Previous reports and design notes
experiments/    Existing exploratory and baseline notebooks
models/         Local model checkpoints
report/         LaTeX report sources and report figures
src/            Reusable Python package for new development
scripts/        Command-line entry points for reproducible workflows
configs/        Versioned experiment and path configuration files
tests/          Lightweight checks for reusable code
artifacts/      Local generated outputs not meant to be edited by hand
```

The existing notebooks in `experiments/` are kept in place for compatibility. New reusable code should go under `src/cough_analysis/`, and new repeatable commands should go under `scripts/`.

The first reusable modules are:

```text
src/cough_analysis/data.py           Metadata loading and raw record decoding
src/cough_analysis/windowing.py      Window indexing and label rules
src/cough_analysis/preprocessing.py  Current Butterworth filtering pipeline
src/cough_analysis/paths.py          Project-relative path helpers
src/cough_analysis/config.py         YAML config loading
```

## Current Dataset

The curated dataset is described by `data/metadata.csv`. Each recording contains four synchronized channels sampled at 4800 Hz:

1. Pulmonary microphone
2. Ambient microphone
3. Stretch sensor with embedded cough label
4. Accelerometer Z-axis

The third channel is decoded with bitwise operations:

```python
cough_label = raw_col3 & 1
stretch_signal = raw_col3 >> 1
```

The current V3 baseline results are documented in `docs/V3BaselineResults.md`.

## Environment

Create a Python environment and install the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

For PyTorch, the exact installation command may depend on the machine and accelerator support. If the generic install fails, install `torch` and `torchaudio` using the command recommended for the target system, then rerun the remaining requirements.

The project targets Python 3.11 or 3.12. Python 3.14 is not recommended for this project because some scientific and ML packages may lag behind newer interpreter releases.

The currently verified package versions are recorded in `requirements-lock.txt`. To recreate the same package set more strictly:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .[dev]
```

If using conda or mamba:

```bash
conda env create -f environment.yml
conda activate ee492-cough
```

## Reproducibility Direction

The project will use three layers:

- Git for source code, reports, small metadata files, and configuration.
- DVC for large data files, processed datasets, and model checkpoints.
- MLflow for experiment parameters, metrics, and run artifacts.

The current repository is prepared so these tools can be introduced without changing the existing notebooks first.

## Common Commands

Compile Python files:

```bash
make compile
```

Check the local environment and dataset paths:

```bash
make env-check
```

Run lightweight tests:

```bash
make test
```

Run a quick V3 model dry-run:

```bash
PYTHONPATH=src python scripts/train_v3.py --dry-run
```

Train the V3 cough detector:

```bash
PYTHONPATH=src python scripts/train_v3.py --config configs/v3.yaml --output artifacts/models/v3_cough.pt
```

Log a V3 training run with MLflow:

```bash
make train-v3-mlflow PYTHON=.venv/bin/python TRAIN_ARGS="--config configs/v3.yaml --output artifacts/models/v3_cough.pt"
```

Evaluate a saved V3 checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate_v3.py --checkpoint artifacts/models/v3_cough.pt --split test
```

Log a V3 evaluation run with MLflow:

```bash
make eval-v3-mlflow PYTHON=.venv/bin/python EVAL_ARGS="--split test"
```

Start the local MLflow UI:

```bash
make mlflow-ui PYTHON=.venv/bin/python
```

Generate report confusion matrix figures:

```bash
python report/generate_figures.py
```

Rebuild curated data from raw CSV files:

```bash
python scripts/prepare_dataset.py --raw-root data/raw_csv --overwrite
```

The `--overwrite` flag is required because rebuilding curated data replaces existing curated CSV files and `data/metadata.csv`.
