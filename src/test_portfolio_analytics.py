from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.portfolio_analytics import (
    analyze_portfolio,
    process_portfolio_batch,
    process_portfolio_csv,
    score_portfolio,
)
from src.preprocessing.dataset_loader import load_dataset


def _build_valid_batch_input(row_count=5):
    credit_df, _ = load_dataset("credit_risk")
    fraud_df, _ = load_dataset("fraud_detection")

    credit_sample = credit_df.head(row_count).copy().reset_index(drop=True)
    fraud_sample = fraud_df[["Amount", "Time"]].head(row_count).copy().reset_index(drop=True)
    return pd.concat([credit_sample, fraud_sample], axis=1)


def test_batch_scoring_valid_csv():
    batch_df = _build_valid_batch_input(5)
    batch_result = process_portfolio_batch(batch_df)
    scored_df = batch_result["scored_portfolio"]

    assert len(scored_df) == 5
    assert set([
        "credit_probability",
        "credit_score",
        "credit_category",
        "fraud_probability",
        "fraud_score",
        "fraud_category",
        "final_decision",
        "scoring_status",
        "scoring_error",
    ]).issubset(scored_df.columns)
    assert set(scored_df["scoring_status"]) == {"scored"}
    assert batch_result["summary_metrics"]["success_rows"] == 5
    assert batch_result["validation_failures"].empty


def test_batch_scoring_missing_column_csv():
    batch_df = _build_valid_batch_input(3).drop(columns=["purpose"])
    batch_result = process_portfolio_batch(batch_df)
    scored_df = batch_result["scored_portfolio"]

    assert len(scored_df) == 3
    assert set(scored_df["scoring_status"]) == {"validation_failed"}
    assert scored_df["scoring_error"].str.contains("missing required columns").all()
    assert len(batch_result["validation_failures"]) == 3
    assert batch_result["summary_metrics"]["validation_failed_rows"] == 3


def test_batch_scoring_extra_columns_csv():
    batch_df = _build_valid_batch_input(3)
    batch_df["extra_notes"] = ["a", "b", "c"]
    scored_df = score_portfolio(batch_df)

    assert len(scored_df) == 3
    assert set(scored_df["scoring_status"]) == {"scored"}
    assert "extra_notes" in scored_df.columns


def test_batch_scoring_invalid_categorical_value():
    batch_df = _build_valid_batch_input(3)
    batch_df.loc[0, "housing"] = "invalid_housing_value"
    batch_result = process_portfolio_batch(batch_df)
    scored_df = batch_result["scored_portfolio"]

    assert len(scored_df) == 3
    assert set(scored_df["scoring_status"]) == {"validation_failed", "scored"}
    assert batch_result["summary_metrics"]["validation_failed_rows"] == 1
    assert batch_result["summary_metrics"]["success_rows"] == 2


def test_batch_scoring_empty_csv():
    empty_df = pd.DataFrame()

    try:
        score_portfolio(empty_df)
    except ValueError as exc:
        assert "must contain at least one row" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty batch input.")


def test_batch_scoring_with_1000_rows():
    credit_df, _ = load_dataset("credit_risk")
    batch_df = credit_df.head(1000).copy()
    scored_df = score_portfolio(batch_df)
    analytics = analyze_portfolio(scored_df)

    assert len(scored_df) == 1000
    assert analytics["summary_metrics"]["total_rows"] == 1000
    assert analytics["summary_metrics"]["success_rows"] == 1000
    assert analytics["summary_metrics"]["validation_failed_rows"] == 0
    assert analytics["summary_metrics"]["inference_failed_rows"] == 0


def test_batch_scoring_with_separate_fraud_dataframe():
    credit_df, _ = load_dataset("credit_risk")
    fraud_df, _ = load_dataset("fraud_detection")

    credit_batch = credit_df.head(4).copy().reset_index(drop=True)
    fraud_batch = fraud_df[["Amount", "Time"]].head(4).copy().reset_index(drop=True)
    batch_result = process_portfolio_batch(credit_batch, fraud_input_df=fraud_batch)

    assert batch_result["summary_metrics"]["success_rows"] == 4
    assert "High Fraud Risk" in batch_result["fraud_distribution"]


def test_process_portfolio_csv_writes_outputs():
    batch_df = _build_valid_batch_input(6)
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        credit_csv_path = temp_dir_path / "portfolio_credit.csv"
        output_dir = temp_dir_path / "artifacts"
        batch_df.to_csv(credit_csv_path, index=False)

        result = process_portfolio_csv(
            str(credit_csv_path),
            output_dir=output_dir,
            chunk_size=2,
        )
        scored_output = pd.read_csv(result["scored_portfolio_path"])
        validation_output = pd.read_csv(result["validation_failures_path"])

        assert len(scored_output) == 6
        assert validation_output.empty
        assert result["summary_metrics"]["total_rows"] == 6


if __name__ == "__main__":
    test_batch_scoring_valid_csv()
    test_batch_scoring_missing_column_csv()
    test_batch_scoring_extra_columns_csv()
    test_batch_scoring_invalid_categorical_value()
    test_batch_scoring_empty_csv()
    test_batch_scoring_with_1000_rows()
    test_batch_scoring_with_separate_fraud_dataframe()
    test_process_portfolio_csv_writes_outputs()
