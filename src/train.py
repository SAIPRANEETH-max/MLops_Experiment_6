from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_model(preprocessor, X_train, y_train, n_estimators=100, random_state=42):
    """Fit preprocessing and Random Forest together to prevent leakage."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
