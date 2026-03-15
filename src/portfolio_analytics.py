from dataclasses import dataclass, field

import pandas as pd

from src.common.logging_utils import get_logger
from src.common.model_utils import resolve_repo_path
from src.common.validation import validate_record_schema
from src.credit_risk.credit_scoring import get_credit_input_schema
from src.decision_engine import generate_final_decision
from src.fraud_detection.fraud_scoring import get_fraud_input_schema


logger = get_logger(__name__)
SCORING_COLUMNS = [
    "source_row_index",
    "credit_probability",
    "credit_score",
    "credit_category",
    "credit_decision",
    "fraud_probability",
    "fraud_score",
    "fraud_category",
    "fraud_decision",
    "final_decision",
    "scoring_status",
    "scoring_error",
    "credit_latency_ms",
    "fraud_latency_ms",
    "decision_latency_ms",
]
VALIDATION_REPORT_SUFFIX_COLUMNS = [
    "source_row_index",
    "validation_status",
    "validation_error",
]


@dataclass
class PortfolioMetricsAccumulator:
    total_rows: int = 0
    success_rows: int = 0
    validation_failed_rows: int = 0
    inference_failed_rows: int = 0
    approve_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    high_fraud_count: int = 0
    fraud_unavailable_count: int = 0
    credit_score_sum: float = 0.0
    credit_score_count: int = 0
    fraud_score_sum: float = 0.0
    fraud_score_count: int = 0
    approval_rejection_counts: dict = field(
        default_factory=lambda: {"Approve": 0, "Review": 0, "Reject": 0}
    )
    fraud_distribution: dict = field(
        default_factory=lambda: {
            "Low Fraud Risk": 0,
            "Medium Fraud Risk": 0,
            "High Fraud Risk": 0,
            "Unavailable": 0,
        }
    )
    credit_risk_distribution: dict = field(
        default_factory=lambda: {
            "Low Risk": 0,
            "Medium Risk": 0,
            "High Risk": 0,
        }
    )

    def update(self, scored_df):
        self.total_rows += int(len(scored_df))
        self.success_rows += int((scored_df["scoring_status"] == "scored").sum())
        self.validation_failed_rows += int(
            (scored_df["scoring_status"] == "validation_failed").sum()
        )
        self.inference_failed_rows += int(
            (scored_df["scoring_status"] == "inference_failed").sum()
        )

        scored_only_df = scored_df[scored_df["scoring_status"] == "scored"].copy()
        if scored_only_df.empty:
            return

        decision_counts = (
            scored_only_df["final_decision"]
            .value_counts()
            .reindex(["Approve", "Review", "Reject"], fill_value=0)
            .to_dict()
        )
        for decision_name, decision_count in decision_counts.items():
            self.approval_rejection_counts[decision_name] += int(decision_count)

        credit_distribution = (
            scored_only_df["credit_category"]
            .value_counts()
            .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
            .to_dict()
        )
        for category_name, category_count in credit_distribution.items():
            self.credit_risk_distribution[category_name] += int(category_count)

        fraud_distribution = (
            scored_only_df["fraud_category"]
            .fillna("Unavailable")
            .value_counts()
            .reindex(
                ["Low Fraud Risk", "Medium Fraud Risk", "High Fraud Risk", "Unavailable"],
                fill_value=0,
            )
            .to_dict()
        )
        for category_name, category_count in fraud_distribution.items():
            self.fraud_distribution[category_name] += int(category_count)

        credit_scores = pd.to_numeric(scored_only_df["credit_score"], errors="coerce").dropna()
        fraud_scores = pd.to_numeric(scored_only_df["fraud_score"], errors="coerce").dropna()

        self.credit_score_sum += float(credit_scores.sum())
        self.credit_score_count += int(len(credit_scores))
        self.fraud_score_sum += float(fraud_scores.sum())
        self.fraud_score_count += int(len(fraud_scores))

        self.approve_count = self.approval_rejection_counts["Approve"]
        self.review_count = self.approval_rejection_counts["Review"]
        self.reject_count = self.approval_rejection_counts["Reject"]
        self.high_fraud_count = self.fraud_distribution["High Fraud Risk"]
        self.fraud_unavailable_count = self.fraud_distribution["Unavailable"]

    def to_summary_metrics(self):
        return {
            "total_rows": self.total_rows,
            "success_rows": self.success_rows,
            "validation_failed_rows": self.validation_failed_rows,
            "inference_failed_rows": self.inference_failed_rows,
            "approve_count": self.approve_count,
            "review_count": self.review_count,
            "reject_count": self.reject_count,
            "high_fraud_count": self.high_fraud_count,
            "fraud_unavailable_count": self.fraud_unavailable_count,
            "mean_credit_score": (
                round(self.credit_score_sum / self.credit_score_count, 2)
                if self.credit_score_count
                else None
            ),
            "mean_fraud_score": (
                round(self.fraud_score_sum / self.fraud_score_count, 2)
                if self.fraud_score_count
                else None
            ),
            "approval_rate": (
                round((self.approve_count / self.total_rows) * 100, 2)
                if self.total_rows
                else 0.0
            ),
            "rejection_rate": (
                round((self.reject_count / self.total_rows) * 100, 2)
                if self.total_rows
                else 0.0
            ),
        }


def _ensure_portfolio_dataframe(input_data, input_name="Portfolio input"):
    if isinstance(input_data, pd.DataFrame):
        portfolio_df = input_data.copy()
    elif isinstance(input_data, pd.Series):
        portfolio_df = input_data.to_frame().T
    elif isinstance(input_data, dict):
        portfolio_df = pd.DataFrame([input_data])
    else:
        raise TypeError(f"{input_name} must be a pandas DataFrame, Series, or dict.")

    if portfolio_df.empty:
        raise ValueError(f"{input_name} must contain at least one row.")

    return portfolio_df.reset_index(drop=True)


def _resolve_fraud_dataframe(credit_df, fraud_df=None):
    fraud_schema = get_fraud_input_schema()
    required_fraud_columns = fraud_schema["required_columns"]

    if fraud_df is not None:
        fraud_portfolio_df = _ensure_portfolio_dataframe(fraud_df, input_name="Fraud portfolio input")
        if len(fraud_portfolio_df) != len(credit_df):
            raise ValueError(
                "Credit and fraud batch inputs must contain the same number of rows."
            )
        return fraud_portfolio_df, "separate"

    present_fraud_columns = [
        column_name for column_name in required_fraud_columns if column_name in credit_df.columns
    ]
    if len(present_fraud_columns) == len(required_fraud_columns):
        return credit_df.copy(), "combined"
    if present_fraud_columns:
        return credit_df.copy(), "partial"
    return None, "absent"


def _merge_row_payloads(credit_row_payload, fraud_row_payload=None):
    merged_payload = credit_row_payload.copy()
    if fraud_row_payload is not None:
        merged_payload.update(fraud_row_payload)
    return merged_payload


def _filter_row_payload(row_payload, required_columns):
    return {
        column_name: row_payload[column_name]
        for column_name in required_columns
        if column_name in row_payload
    }


def _build_success_row(row_index, row_payload, result):
    credit = result["credit"]
    fraud = result["fraud"]
    fraud_is_available = fraud.get("fraud_status") != "unavailable"

    return {
        **row_payload,
        "source_row_index": row_index,
        "credit_probability": credit["credit_probability"],
        "credit_score": credit["credit_score"],
        "credit_category": credit["credit_category"],
        "credit_decision": credit["credit_decision"],
        "fraud_probability": fraud["fraud_probability"] if fraud_is_available else pd.NA,
        "fraud_score": fraud["fraud_score"] if fraud_is_available else pd.NA,
        "fraud_category": fraud["fraud_category"],
        "fraud_decision": fraud["fraud_decision"],
        "final_decision": result["final_decision"],
        "scoring_status": "scored",
        "scoring_error": pd.NA,
        "credit_latency_ms": credit["credit_latency_ms"],
        "fraud_latency_ms": fraud["fraud_latency_ms"] if fraud_is_available else pd.NA,
        "decision_latency_ms": result["decision_latency_ms"],
    }


def _build_failed_row(row_index, row_payload, status, error_message):
    return {
        **row_payload,
        "source_row_index": row_index,
        "credit_probability": pd.NA,
        "credit_score": pd.NA,
        "credit_category": pd.NA,
        "credit_decision": pd.NA,
        "fraud_probability": pd.NA,
        "fraud_score": pd.NA,
        "fraud_category": pd.NA,
        "fraud_decision": pd.NA,
        "final_decision": pd.NA,
        "scoring_status": status,
        "scoring_error": error_message,
        "credit_latency_ms": pd.NA,
        "fraud_latency_ms": pd.NA,
        "decision_latency_ms": pd.NA,
    }


def _build_validation_failure_row(row_index, row_payload, error_message):
    return {
        **row_payload,
        "source_row_index": row_index,
        "validation_status": "validation_failed",
        "validation_error": error_message,
    }


def _validate_batch_row(credit_row_payload, credit_schema, fraud_row_payload=None, fraud_schema=None, fraud_mode="absent"):
    validation_errors = validate_record_schema(
        credit_row_payload,
        required_columns=credit_schema["required_columns"],
        numerical_columns=credit_schema["numerical_columns"],
        categorical_allowed_values=credit_schema["categorical_allowed_values"],
        input_name="Credit input",
    )

    if fraud_mode == "partial" and fraud_schema is not None:
        missing_fraud_columns = [
            column_name
            for column_name in fraud_schema["required_columns"]
            if column_name not in credit_row_payload
        ]
        validation_errors.append(
            f"Fraud input is missing required columns: {missing_fraud_columns}"
        )
        return validation_errors

    if fraud_row_payload is None or fraud_schema is None:
        return validation_errors

    validation_errors.extend(
        validate_record_schema(
            fraud_row_payload,
            required_columns=fraud_schema["required_columns"],
            numerical_columns=fraud_schema["numerical_columns"],
            categorical_allowed_values=fraud_schema["categorical_allowed_values"],
            input_name="Fraud input",
        )
    )
    return validation_errors


def process_portfolio_batch(credit_input_df, fraud_input_df=None, row_index_start=0):
    credit_df = _ensure_portfolio_dataframe(credit_input_df, input_name="Credit portfolio input")
    fraud_df, fraud_mode = _resolve_fraud_dataframe(credit_df, fraud_input_df)

    credit_schema = get_credit_input_schema()
    fraud_schema = get_fraud_input_schema()

    scored_rows = []
    validation_failures = []

    for local_index in range(len(credit_df)):
        source_row_index = row_index_start + local_index
        credit_row_payload = credit_df.iloc[local_index].to_dict()

        if fraud_mode in {"separate", "combined", "partial"}:
            fraud_row_payload = fraud_df.iloc[local_index].to_dict()
        else:
            fraud_row_payload = None

        merged_payload = _merge_row_payloads(credit_row_payload, fraud_row_payload)
        validation_errors = _validate_batch_row(
            credit_row_payload,
            credit_schema,
            fraud_row_payload=fraud_row_payload,
            fraud_schema=fraud_schema,
            fraud_mode=fraud_mode,
        )
        if validation_errors:
            error_message = "; ".join(validation_errors)
            scored_rows.append(
                _build_failed_row(
                    source_row_index,
                    merged_payload,
                    status="validation_failed",
                    error_message=error_message,
                )
            )
            validation_failures.append(
                _build_validation_failure_row(
                    source_row_index,
                    merged_payload,
                    error_message,
                )
            )
            continue

        filtered_credit_payload = _filter_row_payload(
            credit_row_payload,
            credit_schema["required_columns"],
        )
        credit_row_df = pd.DataFrame([filtered_credit_payload])
        fraud_row_df = None
        if fraud_mode in {"separate", "combined"}:
            filtered_fraud_payload = _filter_row_payload(
                fraud_row_payload,
                fraud_schema["required_columns"],
            )
            fraud_row_df = pd.DataFrame([filtered_fraud_payload])

        try:
            result = generate_final_decision(credit_row_df, fraud_row_df)
        except Exception as exc:
            error_message = str(exc)
            logger.error(
                "Batch inference failed for row_index=%s with error=%s",
                source_row_index,
                error_message,
            )
            scored_rows.append(
                _build_failed_row(
                    source_row_index,
                    merged_payload,
                    status="inference_failed",
                    error_message=error_message,
                )
            )
            continue

        scored_rows.append(_build_success_row(source_row_index, merged_payload, result))

    scored_df = pd.DataFrame(scored_rows)
    validation_failures_df = pd.DataFrame(validation_failures)
    if validation_failures_df.empty:
        validation_failures_df = pd.DataFrame(
            columns=_build_empty_validation_report_columns(
                credit_df.columns.tolist(),
                fraud_df.columns.tolist() if fraud_df is not None and fraud_mode == "separate" else None,
            )
        )

    accumulator = PortfolioMetricsAccumulator()
    accumulator.update(scored_df)
    return {
        "summary_metrics": accumulator.to_summary_metrics(),
        "approval_rejection_counts": accumulator.approval_rejection_counts.copy(),
        "fraud_distribution": accumulator.fraud_distribution.copy(),
        "credit_risk_distribution": accumulator.credit_risk_distribution.copy(),
        "scored_portfolio": scored_df,
        "validation_failures": validation_failures_df,
    }


def score_portfolio(credit_input_df, fraud_input_df=None, row_index_start=0):
    return process_portfolio_batch(
        credit_input_df,
        fraud_input_df=fraud_input_df,
        row_index_start=row_index_start,
    )["scored_portfolio"]


def _ensure_scored_portfolio_dataframe(input_df):
    portfolio_df = _ensure_portfolio_dataframe(input_df)
    if all(column in portfolio_df.columns for column in SCORING_COLUMNS):
        return portfolio_df.copy()
    return score_portfolio(portfolio_df)


def analyze_portfolio(input_df):
    scored_df = _ensure_scored_portfolio_dataframe(input_df)
    accumulator = PortfolioMetricsAccumulator()
    accumulator.update(scored_df)

    return {
        "summary_metrics": accumulator.to_summary_metrics(),
        "approval_rejection_counts": accumulator.approval_rejection_counts.copy(),
        "fraud_distribution": accumulator.fraud_distribution.copy(),
        "credit_risk_distribution": accumulator.credit_risk_distribution.copy(),
        "scored_portfolio": scored_df,
        "validation_failures": scored_df[scored_df["scoring_status"] == "validation_failed"].copy(),
    }


def iter_portfolio_csv_chunks(credit_csv_path, fraud_csv_path=None, chunk_size=1000):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    resolved_credit_path = resolve_repo_path(credit_csv_path)
    credit_chunk_iterator = pd.read_csv(resolved_credit_path, chunksize=chunk_size)

    fraud_chunk_iterator = None
    if fraud_csv_path is not None:
        resolved_fraud_path = resolve_repo_path(fraud_csv_path)
        fraud_chunk_iterator = pd.read_csv(resolved_fraud_path, chunksize=chunk_size)

    for credit_chunk in credit_chunk_iterator:
        fraud_chunk = None
        if fraud_chunk_iterator is not None:
            try:
                fraud_chunk = next(fraud_chunk_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "Fraud CSV contains fewer rows than credit CSV."
                ) from exc

            if len(fraud_chunk) != len(credit_chunk):
                raise ValueError(
                    "Credit and fraud CSV chunks must contain the same number of rows."
                )

        yield credit_chunk.reset_index(drop=True), (
            fraud_chunk.reset_index(drop=True) if fraud_chunk is not None else None
        )

    if fraud_chunk_iterator is not None:
        try:
            next(fraud_chunk_iterator)
        except StopIteration:
            return
        raise ValueError("Fraud CSV contains more rows than credit CSV.")


def _append_dataframe_to_csv(dataframe, output_path, write_header):
    dataframe.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def _build_empty_validation_report_columns(credit_columns, fraud_columns=None):
    combined_columns = list(credit_columns)
    if fraud_columns is not None:
        for column_name in fraud_columns:
            if column_name not in combined_columns:
                combined_columns.append(column_name)
    return combined_columns + VALIDATION_REPORT_SUFFIX_COLUMNS


def process_portfolio_csv(credit_csv_path, output_dir, fraud_csv_path=None, chunk_size=1000):
    output_dir_path = resolve_repo_path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    scored_output_path = output_dir_path / "scored_portfolio.csv"
    validation_output_path = output_dir_path / "validation_failures.csv"

    accumulator = PortfolioMetricsAccumulator()
    wrote_scored_output = False
    wrote_validation_output = False
    empty_validation_columns = None
    row_index_start = 0

    for credit_chunk, fraud_chunk in iter_portfolio_csv_chunks(
        credit_csv_path,
        fraud_csv_path=fraud_csv_path,
        chunk_size=chunk_size,
    ):
        batch_result = process_portfolio_batch(
            credit_chunk,
            fraud_input_df=fraud_chunk,
            row_index_start=row_index_start,
        )
        scored_chunk = batch_result["scored_portfolio"]
        validation_chunk = batch_result["validation_failures"]

        _append_dataframe_to_csv(
            scored_chunk,
            scored_output_path,
            write_header=not wrote_scored_output,
        )
        wrote_scored_output = True

        if empty_validation_columns is None:
            empty_validation_columns = _build_empty_validation_report_columns(
                credit_chunk.columns.tolist(),
                fraud_chunk.columns.tolist() if fraud_chunk is not None else None,
            )

        if not validation_chunk.empty:
            _append_dataframe_to_csv(
                validation_chunk,
                validation_output_path,
                write_header=not wrote_validation_output,
            )
            wrote_validation_output = True

        accumulator.update(scored_chunk)
        row_index_start += len(credit_chunk)

    if row_index_start == 0:
        raise ValueError("Input CSV must contain at least one row.")

    if not wrote_validation_output:
        pd.DataFrame(columns=empty_validation_columns).to_csv(
            validation_output_path,
            index=False,
        )

    return {
        "summary_metrics": accumulator.to_summary_metrics(),
        "approval_rejection_counts": accumulator.approval_rejection_counts.copy(),
        "fraud_distribution": accumulator.fraud_distribution.copy(),
        "credit_risk_distribution": accumulator.credit_risk_distribution.copy(),
        "scored_portfolio_path": str(scored_output_path),
        "validation_failures_path": str(validation_output_path),
        "output_dir": str(output_dir_path),
    }
