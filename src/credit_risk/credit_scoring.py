import logging
import time

from src.common.logging_utils import get_logger
from src.common.model_utils import get_positive_class_probability, load_model_bundle
from src.common.schemas import CreditRiskResult
from src.common.validation import (
    align_features,
    ensure_dataframe,
    get_preprocessor_feature_metadata,
    get_required_columns_from_preprocessor,
    require_non_empty_dataframe,
    require_single_row_dataframe,
    warn_on_unseen_categories,
)

MODEL_PATH = "models/credit_model.pkl"
logger = get_logger(__name__)


def get_required_credit_columns():
    _, preprocessor = load_credit_model()
    return get_required_columns_from_preprocessor(preprocessor)


def get_credit_input_schema():
    _, preprocessor = load_credit_model()
    return get_preprocessor_feature_metadata(preprocessor)


def load_credit_model():
    model, preprocessor = load_model_bundle(MODEL_PATH)
    return model, preprocessor


def _prepare_credit_input(input_df, preprocessor):
    credit_input = ensure_dataframe(input_df, input_name="Credit input")
    require_non_empty_dataframe(credit_input, input_name="Credit input")
    require_single_row_dataframe(credit_input, input_name="Credit input")

    required_columns = get_required_columns_from_preprocessor(preprocessor)
    aligned_credit_input = align_features(
        credit_input,
        required_columns,
        input_name="Credit input",
    )
    warn_on_unseen_categories(
        aligned_credit_input,
        preprocessor,
        input_name="Credit input",
    )
    return aligned_credit_input


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
    start_time = time.perf_counter()
    model, preprocessor = load_credit_model()
    credit_input = _prepare_credit_input(input_df, preprocessor)

    try:
        X_processed = preprocessor.transform(credit_input)
    except Exception as exc:
        logger.error("Credit preprocessing failed: %s", exc)
        raise RuntimeError(f"Credit preprocessing failed: {exc}") from exc

    # Positive class (1) = default.
    prob_default = get_positive_class_probability(
        model, X_processed, positive_class=1, model_name="credit model"
    )

    score = calculate_risk_score(prob_default)
    category = assign_risk_category(score)
    decision = make_credit_decision(category)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = CreditRiskResult(
        credit_probability=round(float(prob_default), 4),
        credit_score=score,
        credit_category=category,
        credit_decision=decision,
        credit_latency_ms=latency_ms,
    )
    logger.debug(
        "Credit scoring completed with category=%s decision=%s latency_ms=%s",
        category,
        decision,
        latency_ms,
    )
    return result.to_dict()
