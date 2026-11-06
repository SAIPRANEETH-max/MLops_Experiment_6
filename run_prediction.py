import json
from pathlib import Path
from src.predict import predict_new_data

ROOT = Path(__file__).resolve().parent
with open(ROOT / "config" / "settings.json", encoding="utf-8") as f:
    settings = json.load(f)

result = predict_new_data(
    ROOT / settings["model_path"],
    ROOT / settings["new_data_path"],
    ROOT / "outputs" / "new_predictions.csv"
)
print(result.to_string(index=False))
print("Saved outputs/new_predictions.csv")
