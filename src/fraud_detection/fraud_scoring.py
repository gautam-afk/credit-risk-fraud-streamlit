import pandas as pd
from src.common.model_utils import get_positive_class_probability, load_model_bundle
from src.common.schemas import FraudRiskResult
from src.common.validation import require_non_empty_dataframe

MODEL_PATH = "models/fraud_model.pkl"


def load_fraud_model():
    model, preprocessor = load_model_bundle(MODEL_PATH)
    return model, preprocessor


def calculate_fraud_score(prob_fraud):
    """
    Convert fraud probability into 0-100 fraud risk score.
    Higher score = higher fraud risk.
    """
    if not 0 <= prob_fraud <= 1:
        raise ValueError("prob_fraud must be between 0 and 1.")

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

    if "credit_amount" not in input_df.columns:
        raise ValueError(
            "Fraud input must contain 'Amount' or 'credit_amount' column."
        )
    if "month_duration" not in input_df.columns:
        raise ValueError(
            "Fraud input must contain 'Time' or 'month_duration' column."
        )

    amount = input_df["credit_amount"]
    event_time = input_df["month_duration"]
    return pd.DataFrame({"Amount": amount, "Time": event_time})


def predict_fraud_risk(input_df):
    require_non_empty_dataframe(input_df, input_name="Fraud input")

    model, preprocessor = load_fraud_model()
    fraud_input = _prepare_fraud_input(input_df)

    # Apply preprocessing used during training
    X_processed = preprocessor.transform(fraud_input)

    # Positive class (1) = fraud.
    prob_fraud = get_positive_class_probability(
        model, X_processed, positive_class=1, model_name="fraud model"
    )

    score = calculate_fraud_score(prob_fraud)
    category = assign_fraud_category(score)
    decision = make_fraud_decision(category)

    result = FraudRiskResult(
        probability_fraud=round(float(prob_fraud), 4),
        fraud_score=score,
        fraud_category=category,
        fraud_decision=decision,
    )
    return result.to_dict()
