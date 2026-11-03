# NexusRoute-Courier-App

A small Python machine-learning example that trains a Ridge regression model on scikit-learn's built-in diabetes dataset.

The model predicts a disease-progression measure for demonstration purposes. It is not a diagnostic tool and must not be used to make medical decisions.

## Requirements

- Python 3.9 or newer

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py
```

The script prints cross-validation and test metrics, then saves the trained model to `models/diabetes_ridge.joblib`. Generated model files are excluded from Git.
