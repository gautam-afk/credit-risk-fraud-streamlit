from src.preprocessing.dataset_loader import load_dataset
from src.credit_risk.credit_scoring import predict_credit_risk


def test_credit_scoring_smoke():
    df, _ = load_dataset("credit_risk")
    sample = df.head(1)
    result = predict_credit_risk(sample)

    assert set(result.keys()) == {
        "probability_default",
        "risk_score",
        "risk_category",
        "decision",
    }
    assert 0 <= result["probability_default"] <= 1
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert result["decision"] in {"Approve", "Review", "Reject"}


if __name__ == "__main__":
    test_credit_scoring_smoke()
    print("Credit scoring smoke test passed.")
