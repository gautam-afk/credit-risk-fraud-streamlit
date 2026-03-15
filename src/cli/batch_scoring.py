import argparse
import json
import logging
from pathlib import Path

from src.portfolio_analytics import process_portfolio_csv


logger = logging.getLogger(__name__)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Run batch portfolio scoring and save scored outputs."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the credit or combined input CSV.",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where scored outputs will be written.",
    )
    parser.add_argument(
        "--fraud-csv",
        dest="fraud_csv",
        help="Optional path to a separate fraud CSV aligned row-for-row with the input CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=1000,
        help="Number of rows to process per chunk. Default: 1000.",
    )
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        result = process_portfolio_csv(
            args.input_csv,
            output_dir=args.output_dir,
            fraud_csv_path=args.fraud_csv,
            chunk_size=args.chunk_size,
        )
    except Exception as exc:
        logger.error("Batch scoring run failed: %s", exc)
        return 2

    metrics_payload = {
        "summary_metrics": result["summary_metrics"],
        "approval_rejection_counts": result["approval_rejection_counts"],
        "fraud_distribution": result["fraud_distribution"],
        "credit_risk_distribution": result["credit_risk_distribution"],
    }
    metrics_output_path = Path(result["output_dir"]) / "portfolio_metrics.json"
    with metrics_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metrics_payload, output_file, indent=2)

    summary = result["summary_metrics"]
    total_failures = summary["validation_failed_rows"] + summary["inference_failed_rows"]

    logger.info("Rows processed: %s", summary["total_rows"])
    logger.info("Success rows: %s", summary["success_rows"])
    logger.info("Failures count: %s", total_failures)
    logger.info("Scored CSV: %s", result["scored_portfolio_path"])
    logger.info("Validation failures CSV: %s", result["validation_failures_path"])
    logger.info("Metrics JSON: %s", metrics_output_path)

    if summary["success_rows"] == 0:
        logger.error("Batch run completed with full failure. No rows were scored successfully.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
