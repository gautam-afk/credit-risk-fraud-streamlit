from src.credit_risk.credit_scoring import predict_credit_risk
from src.fraud_detection.fraud_scoring import predict_fraud_risk


def generate_final_decision(input_df):

    credit_result = predict_credit_risk(input_df)
    fraud_result = predict_fraud_risk(input_df)

    # Fraud overrides credit
    if fraud_result["fraud_category"] == "High Fraud Risk":
        final_decision = "Reject"
    elif fraud_result["fraud_category"] == "Medium Fraud Risk":
        final_decision = "Review"
    else:
        final_decision = credit_result["decision"]

    return {
        "credit": credit_result,
        "fraud": fraud_result,
        "final_decision": final_decision
    }
