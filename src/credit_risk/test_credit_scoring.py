from src.preprocessing.dataset_loader import load_dataset
from src.credit_risk.credit_scoring import predict_credit_risk


def test_credit_scoring_smoke():
    df, _ = load_dataset("credit_risk")
    sample = df.head(1)
    result = predict_credit_risk(sample)

    assert set(result.keys()) == {
        "credit_probability",
        "credit_score",
        "credit_category",
        "credit_decision",
        "credit_latency_ms",
    }
    assert 0 <= result["credit_probability"] <= 1
    assert 0 <= result["credit_score"] <= 100
    assert result["credit_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert result["credit_decision"] in {"Approve", "Review", "Reject"}
    assert result["credit_latency_ms"] >= 0


if __name__ == "__main__":
    test_credit_scoring_smoke()
