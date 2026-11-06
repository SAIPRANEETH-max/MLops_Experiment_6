# MLOps Experiment 6 — End-to-End Heart Disease ML Pipeline

This project implements the experiment workflow described in the supplied lab report: dataset loading and cleaning, feature/target separation, stratified train/test split, median imputation + StandardScaler, Random Forest training, evaluation, model persistence, plotting, prediction, and reproducibility.

> Educational demonstration only. This model is not a medical diagnostic tool.

## Project structure

```text
ML_Experiment_6/
├── config/settings.json
├── data/heart_disease.csv       # add the dataset here
├── data/new_patients.csv        # example inference rows
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                      # model artifact generated at runtime
├── outputs/                     # metrics, predictions, plot generated
├── main.py
├── evaluation_plots.py
├── run_prediction.py
├── requirements.txt
└── .gitignore
```

## 1. Dataset

Place your CSV at `data/heart_disease.csv`. It must contain the 13 clinical input features and a binary `target` column (0 = no disease, 1 = disease). Common Cleveland/Kaggle heart datasets are supported when their columns use the names:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

If your source has a legacy empty `num` column, the loader drops it. Missing markers such as `?` and `NA` are handled. The dataset is not bundled because the report's exact CSV was not attached with the report document; use the same CSV you used for the lab to reproduce the reported values.

## 2. Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents activation, use Command Prompt:
```cmd
venv\Scripts\activate.bat
```

## 3. Run the end-to-end workflow

From the repository root:

```bash
python main.py
```

Generated files:
- `models/heart_pipeline.pkl`
- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/performance.png`

Expected terminal completion message: `PIPELINE EXECUTED SUCCESSFULLY!`

## 4. Plot and infer on new data

```bash
python evaluation_plots.py
python run_prediction.py
```

Prediction inputs are in `data/new_patients.csv`; output is `outputs/new_predictions.csv`. The new-data CSV must contain the same 13 feature columns in the same names. The saved pipeline applies its own fitted preprocessing.

## 5. Reproducibility

The configuration uses `test_size=0.2`, `random_state=42`, `stratify=y`, and `n_estimators=100`. Running `python main.py` again with the same dataset and package versions should reproduce the same split and model results.

The lab report's sample values are accuracy 0.7869, precision 0.7500, recall 0.9091, F1 0.8219, and ROC-AUC 0.8766. Your values can differ if the underlying CSV differs.

## 6. Push to GitHub

Create an empty GitHub repository named `ML_Experiment_6` (do not initialize with README if using the included one), then:

```bash
git init
git add .
git commit -m "Initial commit - End to End ML Pipeline Experiment 6"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ML_Experiment_6.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Do not commit your virtual environment or generated model/output files; `.gitignore` excludes them. Keep the dataset only if you have permission to redistribute it. If you want the dataset tracked in Git, remove its ignore rule only if you add one later; currently the CSV is not ignored.

## Notes

- Preprocessing is fitted inside the sklearn Pipeline on training data only, avoiding preprocessing leakage.
- A Random Forest does not require feature scaling, but StandardScaler is included to match the lab workflow.
- For this binary classification implementation, a positive target is encoded as 1.
