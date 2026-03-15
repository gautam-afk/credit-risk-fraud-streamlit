import joblib
import logging
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Allow running this file directly: python src/fraud_detection/train_fraud_model.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import build_preprocessor
from src.common.logging_utils import get_logger
from src.common.model_utils import resolve_repo_path


logger = get_logger(__name__)


def train_fraud_model():

    # Load dataset + config
    df, config = load_dataset("fraud_detection")

    feature_columns = config["numerical_features"] + config["categorical_features"]
    X = df[feature_columns]
    y = df[config["target_column"]]

    if y.isna().any():
        raise ValueError("Fraud target column contains missing values.")

    try:
        y = y.astype(int)
    except ValueError as exc:
        raise ValueError("Fraud target column must contain numeric 0/1 labels.") from exc

    unexpected_labels = sorted(set(y.unique()) - {0, 1})
    if unexpected_labels:
        raise ValueError(
            f"Fraud target column must contain only 0/1 labels. Found: {unexpected_labels}"
        )

    # Split before fitting preprocessor to avoid train/test leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(
        config["numerical_features"],
        config["categorical_features"]
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Logistic Regression with class weighting
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_processed, y_train)

    # Evaluate
    y_pred = model.predict(X_test_processed)

    logger.info("Fraud confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info("Fraud classification report:\n%s", classification_report(y_test, y_pred))

    # Save model + preprocessor
    model_path = resolve_repo_path("models/fraud_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)

    logger.info("Fraud model saved successfully to %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_fraud_model()
