from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.validation import require_columns


def build_preprocessor(numerical_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def prepare_data(df, target_column, numerical_features, categorical_features):
    required_columns = numerical_features + categorical_features + [target_column]
    require_columns(df, required_columns, input_name="Dataset")

    X = df[numerical_features + categorical_features]
    y = df[target_column]

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor
