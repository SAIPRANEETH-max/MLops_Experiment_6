from pathlib import Path
import joblib
import pandas as pd

def predict_new_data(model_path, input_path, output_path):
    """Load saved full pipeline and predict rows in a CSV file."""
    model_path, input_path, output_path = map(Path, (model_path, input_path, output_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}. Run python main.py first.")
    if not input_path.exists():
        raise FileNotFoundError(f"New-data CSV not found: {input_path}")
    model = joblib.load(model_path)
    new_data = pd.read_csv(input_path, na_values=["?", "NA", "N/A", ""])
    if "target" in new_data.columns:
        new_data = new_data.drop(columns=["target"])
    predictions = model.predict(new_data)
    result = new_data.copy()
    result["prediction"] = predictions
    result["prediction_label"] = ["Heart disease present" if p == 1 else "No heart disease" for p in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
