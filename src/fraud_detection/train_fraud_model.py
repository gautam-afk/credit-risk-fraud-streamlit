import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import prepare_data


def train_fraud_model():

    # Load dataset + config
    df, config = load_dataset("fraud_detection")

    # Prepare data
    X, y, preprocessor = prepare_data(
        df,
        config["target_column"],
        config["numerical_features"],
        config["categorical_features"]
    )

    # Stratified split (VERY IMPORTANT for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression with class weighting
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + preprocessor
    joblib.dump((model, preprocessor), "models/fraud_model.pkl")

    print("\nFraud model saved successfully!")


if __name__ == "__main__":
    train_fraud_model()
