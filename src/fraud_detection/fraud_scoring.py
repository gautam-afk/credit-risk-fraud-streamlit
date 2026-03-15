import logging
import time

from src.common.logging_utils import get_logger
from src.common.model_utils import get_positive_class_probability, load_model_bundle
from src.common.schemas import FraudRiskResult
from src.common.validation import (
    align_features,
    ensure_dataframe,
    get_preprocessor_feature_metadata,
    get_required_columns_from_preprocessor,
    require_non_empty_dataframe,
    require_single_row_dataframe,
    warn_on_unseen_categories,
)

MODEL_PATH = "models/fraud_model.pkl"
logger = get_logger(__name__)


def load_fraud_model():
    model, preprocessor = load_model_bundle(MODEL_PATH)
    return model, preprocessor


def get_required_fraud_columns():
    _, preprocessor = load_fraud_model()
    return get_required_columns_from_preprocessor(preprocessor)


def get_fraud_input_schema():
    _, preprocessor = load_fraud_model()
    return get_preprocessor_feature_metadata(preprocessor)


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


def _prepare_fraud_input(input_df, preprocessor):
    fraud_input = ensure_dataframe(input_df, input_name="Fraud input")
    require_non_empty_dataframe(fraud_input, input_name="Fraud input")
    require_single_row_dataframe(fraud_input, input_name="Fraud input")

    required_columns = get_required_columns_from_preprocessor(preprocessor)
    aligned_fraud_input = align_features(
        fraud_input,
        required_columns,
        input_name="Fraud input",
    )
    warn_on_unseen_categories(
        aligned_fraud_input,
        preprocessor,
        input_name="Fraud input",
    )
    return aligned_fraud_input


def predict_fraud_risk(input_df):
    start_time = time.perf_counter()
    model, preprocessor = load_fraud_model()
    fraud_input = _prepare_fraud_input(input_df, preprocessor)

    try:
        X_processed = preprocessor.transform(fraud_input)
    except Exception as exc:
        logger.error("Fraud preprocessing failed: %s", exc)
        raise RuntimeError(f"Fraud preprocessing failed: {exc}") from exc

    # Positive class (1) = fraud.
    prob_fraud = get_positive_class_probability(
        model, X_processed, positive_class=1, model_name="fraud model"
    )

    score = calculate_fraud_score(prob_fraud)
    category = assign_fraud_category(score)
    decision = make_fraud_decision(category)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = FraudRiskResult(
        fraud_probability=round(float(prob_fraud), 4),
        fraud_score=score,
        fraud_category=category,
        fraud_decision=decision,
        fraud_latency_ms=latency_ms,
    )
    logger.debug(
        "Fraud scoring completed with category=%s decision=%s latency_ms=%s",
        category,
        decision,
        latency_ms,
    )
    return result.to_dict()
