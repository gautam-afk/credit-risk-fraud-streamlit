import joblib
import pandas as pd

MODEL_PATH = "models/fraud_model.pkl"


def load_fraud_model():
    model, preprocessor = joblib.load(MODEL_PATH)
    return model, preprocessor


def calculate_fraud_score(prob_fraud):
    """
    Convert fraud probability into 0-100 fraud risk score.
    Higher score = higher fraud risk.
    """
    score = prob_fraud * 100
    return round(score, 2)


def assign_fraud_category(score):
    if score >= 70:
        return "High Fraud Risk"
    elif score >= 40:
        return "Medium Fraud Risk"
    else:
        return "Low Fraud Risk"


def make_fraud_decision(category):
    if category == "High Fraud Risk":
        return "Reject"
    elif category == "Medium Fraud Risk":
        return "Review"
    else:
        return "Clear"


def _prepare_fraud_input(input_df):
    """
    Align incoming payload with fraud model schema.
    Expected fraud features are Amount and Time.
    """
    if "Amount" in input_df.columns and "Time" in input_df.columns:
        return input_df

    amount = input_df["credit_amount"] if "credit_amount" in input_df.columns else 0
    event_time = input_df["month_duration"] if "month_duration" in input_df.columns else 0

    return pd.DataFrame({"Amount": amount, "Time": event_time})


def predict_fraud_risk(input_df):
    model, preprocessor = load_fraud_model()
    fraud_input = _prepare_fraud_input(input_df)

    # Apply preprocessing used during training
    X_processed = preprocessor.transform(fraud_input)

    # Probability of fraud = class 1
    prob_fraud = float(model.predict_proba(X_processed)[0][1])

    score = calculate_fraud_score(prob_fraud)
    category = assign_fraud_category(score)
    decision = make_fraud_decision(category)

    return {
        "probability_fraud": round(float(prob_fraud), 4),
        "fraud_score": score,
        "fraud_category": category,
        "fraud_decision": decision
    }
