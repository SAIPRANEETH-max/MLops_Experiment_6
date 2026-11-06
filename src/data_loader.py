from pathlib import Path
import pandas as pd

def load_data(path):
    """Load CSV and normalize common missing-value markers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place heart_disease.csv in the data/ folder. "
            "See README.md for dataset guidance."
        )
    df = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])
    # Some Cleveland-derived datasets contain an empty legacy 'num' column.
    if "num" in df.columns and df["num"].isna().all():
        df = df.drop(columns=["num"])
    df = df.drop_duplicates().copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "target" not in df.columns:
        # UCI Cleveland files may use 'num' as the disease target.
        if "num" in df.columns:
            df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
            df = df.drop(columns=["num"])
        else:
            raise ValueError("Expected a 'target' column in the dataset.")
    df = df.dropna(subset=["target"])
    df["target"] = (df["target"] > 0).astype(int)
    return df
