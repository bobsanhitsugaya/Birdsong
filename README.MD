# Birdsong ML Pipeline

## Overview
This repository contains a modular and configurable machine learning pipeline for birdsong species classification using audio recordings. The pipeline is implemented in Python and is organized for easy experimentation with different algorithms and parameters.

### Folder Structure
- `src/` : Python modules for data loading, feature extraction, model building, and training
- `config.yaml` : Configuration file for pipeline parameters
- `run.sh` : Bash script to execute the pipeline
- `requirements.txt` : Python dependencies
- `eda.ipynb` : Exploratory Data Analysis (EDA) notebook
- `README.md` : This documentation

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   bash run.sh
   ```
   or directly:
   ```bash
   python -m src.train --config config.yaml
   ```
3. To change parameters (e.g., model, features, data paths), edit `config.yaml`.

## Pipeline Flow
1. **Config Loading**: Reads parameters from `config.yaml` (can be overridden by environment variable or CLI).
2. **Data Loading**: Loads metadata and audio files, filters by quality.
3. **Feature Extraction**: Extracts MFCC features from each audio file.
4. **Label Encoding**: Encodes species names as integers.
5. **Data Splitting**: Splits data into train, validation, and test sets.
6. **Model Building**: Builds a 1D CNN (configurable for other models).
7. **Training**: Trains with early stopping and model checkpointing.
8. **Evaluation**: Evaluates on test set and visualizes accuracy/loss.

```
[Data] -> [Feature Extraction] -> [Split] -> [Model] -> [Train] -> [Eval]
```

## Feature Engineering Summary
| Feature         | Description                        | Processing         |
|----------------|------------------------------------|--------------------|
| MFCCs          | Mel Frequency Cepstral Coefficients| Mean aggregation   |
| Quality Filter | Only A/B/C recordings              | Filtered in script |
| Label          | Bird species (English name)        | Label encoded      |

## Model Choice
- **1D CNN**: Chosen for its effectiveness on MFCC features and audio classification tasks. Easily extendable to other models via `src/models.py` and config.

## Evaluation
- **Metrics**: Accuracy (train/val/test), loss curves
- **Early Stopping**: Prevents overfitting
- **Visualization**: Plots accuracy and loss per epoch

## EDA Key Findings (Summary)
- Dataset is imbalanced; some species are rare
- Most recordings are short and high quality
- MFCCs are effective for audio-based classification
- Filtering and label encoding are essential for robust modeling

For detailed EDA, see `eda.ipynb`.

---

**To experiment with different models or parameters, simply edit `config.yaml` and re-run the pipeline.**
