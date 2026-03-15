import joblib
import logging
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
from src.common.logging_utils import get_logger
from src.common.model_utils import resolve_repo_path


TARGET_MAPPING = {
    "good": 0,
    "bad": 1,
}
logger = get_logger(__name__)


def train_credit_model():

    # Load dataset and config
    df, config = load_dataset("credit_risk")
    raw_target = df[config["target_column"]].copy()

    # Encode target so class 1 consistently means default / bad credit.
    df[config["target_column"]] = (
        raw_target
        .astype(str)
        .str.strip()
        .str.lower()
        .map(TARGET_MAPPING)
    )
    if df[config["target_column"]].isna().any():
        unexpected_labels = sorted(
            raw_target.loc[df[config["target_column"]].isna()]
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        raise ValueError(
            "Unexpected target labels found in credit dataset. "
            f"Expected labels: {sorted(TARGET_MAPPING.keys())}. "
            f"Found: {unexpected_labels}"
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

    logger.info("Credit model accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info("Credit classification report:\n%s", classification_report(y_test, y_pred))

    # Save model + preprocessor
    model_path = resolve_repo_path("models/credit_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)

    logger.info("Credit model saved successfully to %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train_credit_model()
