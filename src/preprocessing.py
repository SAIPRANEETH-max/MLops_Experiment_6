from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessor(X):
    """Build numeric preprocessing: median imputation followed by scaling."""
    numeric_features = list(X.columns)
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, numeric_features)],
        remainder="drop"
    )
