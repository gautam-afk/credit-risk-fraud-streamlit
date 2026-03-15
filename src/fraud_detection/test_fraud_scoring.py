from src.preprocessing.dataset_loader import load_dataset
from src.fraud_detection.fraud_scoring import predict_fraud_risk


def test_fraud_scoring_smoke():
    df, _ = load_dataset("fraud_detection")
    sample = df.head(1)
    result = predict_fraud_risk(sample)

    assert set(result.keys()) == {
        "fraud_probability",
        "fraud_score",
        "fraud_category",
        "fraud_decision",
        "fraud_latency_ms",
        "fraud_status",
        "fraud_reason",
    }
    assert 0 <= result["fraud_probability"] <= 1
    assert 0 <= result["fraud_score"] <= 100
    assert result["fraud_category"] in {
        "Low Fraud Risk",
        "Medium Fraud Risk",
        "High Fraud Risk",
    }
    assert result["fraud_decision"] in {"Clear", "Review", "Reject"}
    assert result["fraud_latency_ms"] >= 0
    assert result["fraud_status"] == "scored"
    assert result["fraud_reason"] is None


if __name__ == "__main__":
    test_fraud_scoring_smoke()
