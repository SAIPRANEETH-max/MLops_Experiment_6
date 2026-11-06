import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.feature_engineering import prepare_features
from src.preprocessing import build_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model

ROOT = Path(__file__).resolve().parent

def read_settings():
    with open(ROOT / "config" / "settings.json", encoding="utf-8") as file:
        return json.load(file)

def main():
    settings = read_settings()
    data_path = ROOT / settings["dataset_path"]
    print("[1/7] Loading and cleaning data...")
    df = load_data(data_path)
    print(f"Clean dataset shape: {df.shape}; duplicates: {df.duplicated().sum()}")

    print("[2/7] Separating features and target...")
    X, y = prepare_features(df, settings["target_column"])
    print(f"Features: {X.shape}; target distribution: {y.value_counts().to_dict()}")

    print("[3/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings["test_size"],
        random_state=settings["random_state"], stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    print("[4/7] Building preprocessing and training Random Forest...")
    preprocessor = build_preprocessor(X_train)
    model = train_model(
        preprocessor, X_train, y_train,
        n_estimators=settings["n_estimators"],
        random_state=settings["random_state"]
    )

    print("[5/7] Evaluating...")
    metrics, predictions = evaluate_model(model, X_test, y_test)
    output_dir = ROOT / "outputs"
    model_dir = ROOT / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    metrics_path = ROOT / settings["metrics_path"]
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions
    }).to_csv(ROOT / settings["predictions_path"], index=False)

    model_path = ROOT / settings["model_path"]
    joblib.dump(model, model_path)

    print("[6/7] Creating performance plot...")
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Heart Disease Classifier Performance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(ROOT / settings["performance_plot_path"], dpi=150)
    plt.close()

    print("[7/7] Results")
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")
    print(f"Model saved: {model_path.relative_to(ROOT)}")
    print(f"Metrics saved: {metrics_path.relative_to(ROOT)}")
    print("PIPELINE EXECUTED SUCCESSFULLY!")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PIPELINE FAILED: {exc}")
        raise
