
import pandas as pd
from pathlib import Path

# Original UCI Cleveland Heart Disease dataset
url = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Download the complete dataset
df = pd.read_csv(
    url,
    names=columns,
    na_values="?"
)

# Convert original target:
# 0 = no heart disease
# 1, 2, 3, 4 = heart disease present
df["target"] = (df["num"] > 0).astype(int)

# Remove the original target column
df = df.drop(columns=["num"])

# Create data folder if it does not exist
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Save the complete dataset
output_path = data_folder / "heart_disease.csv"
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print("Saved to:", output_path)
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
