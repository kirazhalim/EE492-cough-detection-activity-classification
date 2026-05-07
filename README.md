# Multi-Sensor Cough Analysis and Activity Classification

This repository is my EE491-EE492 senior design project. I worked on cough detection and activity classification with wearable sensor recordings. The main idea is to use audio and motion signals together, instead of using only one sensor.

The project uses 85 short recordings, around 28.33 minutes in total. Each recording has 4 synchronized channels sampled at 4800 Hz:

- Pulmonary microphone
- Ambient microphone
- Stretch sensor
- Accelerometer Z-axis

The stretch sensor channel also stores the cough label in its last bit. I decode it with bitwise operations before training.

## What I Did

- Prepared a reusable Python pipeline under `src/cough_analysis/`.
- Organized the dataset with `data/metadata.csv`.
- Used record-level train, validation, and test splits so the same recording does not appear in different splits.
- Built three deep learning versions for cough detection.
- Added event-level evaluation, not only window-level evaluation.
- Tested activity classification with sitting, standing, and walking classes.
- Kept scripts and configs so the main experiments can be repeated.

## Dataset Summary

The V3 cough detection setup uses a 70/15/15 record-level split:

| Split | Records | Duration (min) | Windows | Cough windows | Cough events |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 59 | 19.67 | 4543 | 788 | 157 |
| Validation | 13 | 4.33 | 1001 | 226 | 43 |
| Test | 13 | 4.33 | 1001 | 192 | 37 |
| All | 85 | 28.33 | 6545 | 1206 | 237 |

Activity distribution in the metadata:

| Activity | Records |
| --- | ---: |
| Sitting | 43 |
| Walking | 22 |
| Standing | 15 |
| Running | 5 |

Running has only 5 records, so I did not use it in the final V3 activity classification setup.

## Method

I first preprocess the four channels:

- Audio channels: 60-2200 Hz bandpass filter
- Stretch sensor: 20 Hz low-pass filter
- Accelerometer: 20 Hz low-pass filter
- Window size: 1.0 second
- V2 and V3 hop size: 0.25 second

For cough detection, I tested three versions:

| Version | Main change |
| --- | --- |
| V1 | Raw waveform input with a 1D CNN |
| V2 | Center-based labels, data augmentation, and learning rate scheduling |
| V3 | Log-Mel spectrogram input with a 2D CNN audio branch |

The model has two branches. One branch uses the two audio channels, and the other branch uses stretch and accelerometer signals. Then the two outputs are joined before the final classifier.

## Main Results

The best cough detection model is V3. On the test split, it reached:

| Metric | Value |
| --- | ---: |
| Test windows | 1001 |
| Window accuracy | 0.941 |
| Cough precision | 0.806 |
| Cough recall | 0.911 |
| Cough F1 | 0.856 |
| Event precision | 0.917 |
| Event recall | 0.892 |
| Event F1 | 0.904 |

At event level, V3 detected 33 out of 37 cough events and produced 3 false positive events.

For activity classification, the best V3 setup used 3 classes: sitting, standing, and walking. It reached 0.92 accuracy and 0.91 macro F1 on the test set.

More detailed results are in `docs/V3BaselineResults.md` and `report/main.pdf`.

## Repository Structure

```text
data/           Metadata and dataset preparation files
docs/           Reports and result notes
experiments/    Earlier notebooks
report/         LaTeX report and figures
src/            Reusable Python package
scripts/        Training, evaluation, and dataset scripts
configs/        Experiment configuration files
tests/          Lightweight tests
artifacts/      Local generated outputs
```

## Setup

Create a Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The project targets Python 3.11 or 3.12.

## Useful Commands

Run tests:

```bash
make test
```

Check the environment and dataset paths:

```bash
make env-check
```

Train the V3 cough detector:

```bash
PYTHONPATH=src python scripts/train_v3.py --config configs/v3.yaml --output artifacts/models/v3_cough.pt
```

Evaluate a V3 checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate_v3.py --checkpoint artifacts/models/v3_cough.pt --split test --threshold 0.6
```

Generate the report figures:

```bash
python report/generate_figures.py
```

## Notes

This project is a prototype for analysis and research. It is not a medical diagnosis tool. The dataset is also small for deep learning, so more subjects and longer recordings would be needed for a stronger real-world evaluation.
