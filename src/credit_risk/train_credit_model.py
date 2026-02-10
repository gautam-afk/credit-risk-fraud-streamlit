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
from src.preprocessing.preprocessor import prepare_data


def train_credit_model():

    # Load dataset and config
    df, config = load_dataset("credit_risk")

    # Encode target (good -> 1, bad -> 0)
    df[config["target_column"]] = df[config["target_column"]].map({
        "good": 1,
        "bad": 0
    })

    # Prepare data
    X, y, preprocessor = prepare_data(
        df,
        config["target_column"],
        config["numerical_features"],
        config["categorical_features"]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model + preprocessor
    joblib.dump(
        (model, preprocessor),
        "models/credit_model.pkl"
    )

    print("Model saved successfully!")


if __name__ == "__main__":
    train_credit_model()
