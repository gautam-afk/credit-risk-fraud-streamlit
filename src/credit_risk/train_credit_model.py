import joblib
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Allow running this file directly: python src/credit_risk/train_credit_model.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import build_preprocessor


def train_credit_model():

    # Load dataset and config
    df, config = load_dataset("credit_risk")

    # Encode target (good -> 1, bad -> 0)
    df[config["target_column"]] = df[config["target_column"]].map({
        "good": 1,
        "bad": 0
    })
    if df[config["target_column"]].isna().any():
        raise ValueError(
            "Unexpected target labels found in credit dataset. "
            "Expected labels: 'good' and 'bad'."
        )

    feature_columns = config["numerical_features"] + config["categorical_features"]
    X = df[feature_columns]
    y = df[config["target_column"]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit preprocessing only on training split to avoid leakage
    preprocessor = build_preprocessor(
        config["numerical_features"],
        config["categorical_features"]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_processed, y_train)

    # Evaluation
    y_pred = model.predict(X_test_processed)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model + preprocessor
    model_path = Path("models/credit_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_credit_model()
