import joblib
from pathlib import Path

MODEL_PATH = "models/credit_model.pkl"


def load_credit_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            "Model file not found at models/credit_model.pkl. Run training first."
        )
    model, preprocessor = joblib.load(MODEL_PATH)
    return model, preprocessor


def calculate_risk_score(prob_default):
    """
    Convert probability of default into a 0-100 risk score.
    Higher score = lower risk.
    """
    if not 0 <= prob_default <= 1:
        raise ValueError("prob_default must be between 0 and 1.")

    score = (1 - prob_default) * 100
    return round(score, 2)


def assign_risk_category(score):
    if score >= 70:
        return "Low Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "High Risk"


def make_credit_decision(category):
    if category == "Low Risk":
        return "Approve"
    elif category == "Medium Risk":
        return "Review"
    else:
        return "Reject"


def predict_credit_risk(input_df):
    model, preprocessor = load_credit_model()

    # Apply same preprocessing used during training
    X_processed = preprocessor.transform(input_df)

    # Probability of default is class 1
    prob_default = model.predict_proba(X_processed)[0][1]

    score = calculate_risk_score(prob_default)
    category = assign_risk_category(score)
    decision = make_credit_decision(category)

    return {
        "probability_default": round(prob_default, 4),
        "risk_score": score,
        "risk_category": category,
        "decision": decision
    }
