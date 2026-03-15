import logging
import time

from src.common.logging_utils import get_logger
from src.common.schemas import FinalDecisionResult
from src.common.validation import ensure_dataframe, require_non_empty_dataframe
from src.credit_risk.credit_scoring import predict_credit_risk
from src.fraud_detection.fraud_scoring import get_required_fraud_columns, predict_fraud_risk


logger = get_logger(__name__)


def _can_score_fraud(input_df):
    required_columns = get_required_fraud_columns()
    return all(column in input_df.columns for column in required_columns)


def generate_final_decision(credit_input_df, fraud_input_df=None):
    start_time = time.perf_counter()
    credit_input_df = ensure_dataframe(credit_input_df, input_name="Credit input")
    require_non_empty_dataframe(credit_input_df, input_name="Credit input")

    if fraud_input_df is None:
        fraud_input_df = credit_input_df
    else:
        fraud_input_df = ensure_dataframe(fraud_input_df, input_name="Fraud input")
        require_non_empty_dataframe(fraud_input_df, input_name="Fraud input")

    credit_result = predict_credit_risk(credit_input_df)

    if _can_score_fraud(fraud_input_df):
        fraud_result = predict_fraud_risk(fraud_input_df)

        # Fraud overrides credit
        if fraud_result["fraud_category"] == "High Fraud Risk":
            final_decision = "Reject"
        elif fraud_result["fraud_category"] == "Medium Fraud Risk":
            final_decision = "Review"
        else:
            final_decision = credit_result["credit_decision"]
    else:
        required_columns = get_required_fraud_columns()
        logger.debug(
            "Fraud scoring skipped because required columns were not present: %s",
            required_columns,
        )
        fraud_result = {
            "fraud_probability": None,
            "fraud_score": None,
            "fraud_category": "Unavailable",
            "fraud_decision": "Unavailable",
            "fraud_latency_ms": None,
            "fraud_status": "unavailable",
            "fraud_reason": (
                "Fraud scoring requires these columns: "
                f"{required_columns}."
            ),
        }
        final_decision = credit_result["credit_decision"]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = FinalDecisionResult(
        credit=credit_result,
        fraud=fraud_result,
        final_decision=final_decision,
        decision_latency_ms=latency_ms,
    )
    logger.debug("Final decision completed with decision=%s latency_ms=%s", final_decision, latency_ms)
    return result.to_dict()
