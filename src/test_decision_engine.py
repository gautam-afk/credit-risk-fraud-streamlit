from src.preprocessing.dataset_loader import load_dataset
from src.decision_engine import generate_final_decision


def test_decision_engine_smoke():
    df, _ = load_dataset("credit_risk")
    sample = df.head(1)
    result = generate_final_decision(sample)

    assert set(result.keys()) == {"credit", "fraud", "final_decision"}
    assert result["final_decision"] in {"Approve", "Review", "Reject"}
    assert "decision" in result["credit"]
    assert "fraud_decision" in result["fraud"]


if __name__ == "__main__":
    test_decision_engine_smoke()
    print("Decision engine smoke test passed.")
