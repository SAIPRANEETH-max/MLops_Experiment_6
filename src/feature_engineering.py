def prepare_features(df, target_column="target"):
    """Separate model inputs and target, and validate feature columns."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(int).copy()
    if X.empty:
        raise ValueError("No input features found.")
    return X, y
