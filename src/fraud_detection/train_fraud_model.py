import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import build_preprocessor


def train_fraud_model():

    # Load dataset + config
    df, config = load_dataset("fraud_detection")

    feature_columns = config["numerical_features"] + config["categorical_features"]
    X = df[feature_columns]
    y = df[config["target_column"]]

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

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + preprocessor
    model_path = Path("models/fraud_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, preprocessor), model_path)

    print("\nFraud model saved successfully!")


if __name__ == "__main__":
    train_fraud_model()
