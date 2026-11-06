"""Regenerate a bar chart from outputs/metrics.json."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
metrics_path = ROOT / "outputs" / "metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError("Run python main.py before plotting metrics.")
with open(metrics_path, encoding="utf-8") as f:
    metrics = json.load(f)
plt.figure(figsize=(9, 5))
plt.bar(list(metrics.keys()), list(metrics.values()))
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Heart Disease Classifier Performance")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "performance.png", dpi=150)
plt.close()
print("Saved outputs/performance.png")
