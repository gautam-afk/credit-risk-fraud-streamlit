from src.common.model_utils import get_positive_class_probability, load_model_bundle
from src.common.schemas import CreditRiskResult
from src.common.validation import get_required_columns_from_preprocessor, require_columns, require_non_empty_dataframe

MODEL_PATH = "models/credit_model.pkl"


def _validate_credit_input(input_df, preprocessor):
    require_non_empty_dataframe(input_df, input_name="Credit input")
    required_columns = get_required_columns_from_preprocessor(preprocessor)
    require_columns(input_df, required_columns, input_name="Credit input")


def load_credit_model():
    model, preprocessor = load_model_bundle(MODEL_PATH)
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
    _validate_credit_input(input_df, preprocessor)

    # Apply same preprocessing used during training
    X_processed = preprocessor.transform(input_df)

    # Positive class (1) = default.
    prob_default = get_positive_class_probability(
        model, X_processed, positive_class=1, model_name="credit model"
    )

    score = calculate_risk_score(prob_default)
    category = assign_risk_category(score)
    decision = make_credit_decision(category)

    result = CreditRiskResult(
        probability_default=round(float(prob_default), 4),
        risk_score=score,
        risk_category=category,
        decision=decision,
    )
    return result.to_dict()
