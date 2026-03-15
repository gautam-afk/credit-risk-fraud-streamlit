from src.preprocessing.dataset_loader import load_dataset
from src.decision_engine import generate_final_decision


def test_decision_engine_credit_only_smoke():
    credit_df, _ = load_dataset("credit_risk")
    credit_sample = credit_df.head(1).copy()
    result = generate_final_decision(credit_sample)

    assert set(result.keys()) == {"credit", "fraud", "final_decision", "decision_latency_ms"}
    assert result["final_decision"] in {"Approve", "Review", "Reject"}
    assert "credit_decision" in result["credit"]
    assert result["fraud"]["fraud_status"] == "unavailable"
    assert result["decision_latency_ms"] >= 0


def test_decision_engine_with_separate_fraud_input():
    credit_df, _ = load_dataset("credit_risk")
    fraud_df, _ = load_dataset("fraud_detection")

    credit_sample = credit_df.head(1).copy()
    fraud_sample = fraud_df[["Amount", "Time"]].head(1).copy()
    result = generate_final_decision(credit_sample, fraud_sample)

    assert set(result["fraud"].keys()) == {
        "fraud_probability",
        "fraud_score",
        "fraud_category",
        "fraud_decision",
        "fraud_latency_ms",
        "fraud_status",
        "fraud_reason",
    }
    assert result["fraud"]["fraud_decision"] in {"Clear", "Review", "Reject"}
    assert result["fraud"]["fraud_status"] == "scored"


if __name__ == "__main__":
    test_decision_engine_credit_only_smoke()
    test_decision_engine_with_separate_fraud_input()
