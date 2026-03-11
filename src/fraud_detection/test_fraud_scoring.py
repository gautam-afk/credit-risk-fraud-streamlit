from src.preprocessing.dataset_loader import load_dataset
from src.fraud_detection.fraud_scoring import predict_fraud_risk


def test_fraud_scoring_smoke():
    df, _ = load_dataset("fraud_detection")
    sample = df.head(1)
    result = predict_fraud_risk(sample)

    assert set(result.keys()) == {
        "probability_fraud",
        "fraud_score",
        "fraud_category",
        "fraud_decision",
    }
    assert 0 <= result["probability_fraud"] <= 1
    assert 0 <= result["fraud_score"] <= 100
    assert result["fraud_category"] in {
        "Low Fraud Risk",
        "Medium Fraud Risk",
        "High Fraud Risk",
    }
    assert result["fraud_decision"] in {"Clear", "Review", "Reject"}


if __name__ == "__main__":
    test_fraud_scoring_smoke()
    print("Fraud scoring smoke test passed.")
