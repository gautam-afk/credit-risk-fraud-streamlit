import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision_engine import generate_final_decision
from src.portfolio_analytics import process_portfolio_batch
from src.preprocessing.dataset_loader import load_dataset

st.set_page_config(page_title="Credit Risk System")

st.title("Credit Risk and Fraud Detection System")

if "portfolio_df" not in st.session_state:
    st.session_state["portfolio_df"] = None
if "portfolio_fraud_df" not in st.session_state:
    st.session_state["portfolio_fraud_df"] = None
if "scored_portfolio_df" not in st.session_state:
    st.session_state["scored_portfolio_df"] = None
if "validation_failures_df" not in st.session_state:
    st.session_state["validation_failures_df"] = None
if "portfolio_metrics" not in st.session_state:
    st.session_state["portfolio_metrics"] = None
if "portfolio_source_signature" not in st.session_state:
    st.session_state["portfolio_source_signature"] = None


@st.cache_data(show_spinner=False)
def load_credit_form_options():
    credit_df, credit_config = load_dataset("credit_risk")
    categorical_options = {}

    for column_name in credit_config["categorical_features"]:
        categorical_options[column_name] = sorted(
            credit_df[column_name].dropna().astype(str).unique().tolist()
        )

    return categorical_options


credit_form_options = load_credit_form_options()
applicant_tab, portfolio_tab = st.tabs(["Applicant Scoring", "Portfolio Analytics"])


@st.cache_data(show_spinner=False)
def convert_dataframe_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")

with applicant_tab:
    st.header("Credit Risk Inputs")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("age", min_value=18, max_value=100, value=30)
        credit_amount = st.number_input("credit_amount", min_value=0.0, value=5000.0)

    with col2:
        month_duration = st.number_input("month_duration", min_value=1.0, value=12.0)
        housing = st.selectbox("housing", credit_form_options["housing"])

    years_employment = st.selectbox(
        "years_employment",
        credit_form_options["years_employment"],
    )

    purpose = st.selectbox(
        "purpose",
        credit_form_options["purpose"],
    )

    st.divider()
    st.header("Fraud Detection Inputs")
    f1, f2 = st.columns(2)

    with f1:
        amount = st.number_input("Amount", min_value=0.0, value=5000.0)
    with f2:
        time = st.number_input("Time", min_value=0.0, value=12.0)

    st.divider()

    if st.button("Evaluate Applicant", key="evaluate_applicant"):
        credit_input = pd.DataFrame([{
            "age": age,
            "credit_amount": credit_amount,
            "month_duration": month_duration,
            "housing": housing,
            "years_employment": years_employment,
            "purpose": purpose,
        }])
        fraud_input = pd.DataFrame([{
            "Amount": amount,
            "Time": time,
        }])

        try:
            result = generate_final_decision(credit_input, fraud_input)
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
        else:
            st.subheader("Final Decision")
            decision = result["final_decision"]

            if decision == "Approve":
                st.success(f"Decision: {decision}")
            elif decision == "Review":
                st.warning(f"Decision: {decision}")
            else:
                st.error(f"Decision: {decision}")

            st.divider()

            st.subheader("Credit Risk Analysis")
            credit = result["credit"]
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Credit Probability", credit["credit_probability"])
            c2.metric("Credit Score", credit["credit_score"])
            c3.metric("Credit Category", credit["credit_category"])
            c4.metric("Credit Latency (ms)", credit["credit_latency_ms"])

            st.divider()

            st.subheader("Fraud Risk Analysis")
            fraud = result["fraud"]
            if fraud.get("fraud_status") == "unavailable":
                st.info(fraud["fraud_reason"])
            else:
                fraud_c1, fraud_c2, fraud_c3, fraud_c4 = st.columns(4)
                fraud_c1.metric("Fraud Probability", fraud["fraud_probability"])
                fraud_c2.metric("Fraud Score", fraud["fraud_score"])
                fraud_c3.metric("Fraud Category", fraud["fraud_category"])
                fraud_c4.metric("Fraud Latency (ms)", fraud["fraud_latency_ms"])

with portfolio_tab:
    st.header("Batch Portfolio Scoring")
    st.caption("Upload a credit CSV, optionally add an aligned fraud CSV, score each row safely, and download the outputs.")

    uploaded_file = st.file_uploader(
        "Credit / Combined Portfolio CSV",
        type=["csv"],
        key="portfolio_credit_csv",
    )
    uploaded_fraud_file = st.file_uploader(
        "Optional Fraud CSV",
        type=["csv"],
        key="portfolio_fraud_csv",
    )
    use_sample_portfolio = st.button("Load Sample Portfolio", key="load_sample_portfolio")

    if uploaded_file is not None:
        fraud_signature = (
            uploaded_fraud_file.name,
            uploaded_fraud_file.size,
        ) if uploaded_fraud_file is not None else None
        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
            fraud_signature,
        )
        if st.session_state["portfolio_source_signature"] != file_signature:
            st.session_state["portfolio_df"] = pd.read_csv(uploaded_file)
            st.session_state["portfolio_fraud_df"] = (
                pd.read_csv(uploaded_fraud_file) if uploaded_fraud_file is not None else None
            )
            st.session_state["scored_portfolio_df"] = None
            st.session_state["validation_failures_df"] = None
            st.session_state["portfolio_metrics"] = None
            st.session_state["portfolio_source_signature"] = file_signature
    elif use_sample_portfolio:
        sample_df, _ = load_dataset("credit_risk")
        st.session_state["portfolio_df"] = sample_df.head(25).copy()
        st.session_state["portfolio_fraud_df"] = None
        st.session_state["scored_portfolio_df"] = None
        st.session_state["validation_failures_df"] = None
        st.session_state["portfolio_metrics"] = None
        st.session_state["portfolio_source_signature"] = "sample_credit_risk_head_25"

    portfolio_df = st.session_state["portfolio_df"]
    portfolio_fraud_df = st.session_state["portfolio_fraud_df"]
    if portfolio_df is not None:
        st.write("Credit / Combined Preview")
        st.dataframe(portfolio_df.head(10), use_container_width=True)
        if portfolio_fraud_df is not None:
            st.write("Fraud Preview")
            st.dataframe(portfolio_fraud_df.head(10), use_container_width=True)

        if st.button("Run Batch Scoring", key="run_batch_scoring"):
            try:
                batch_result = process_portfolio_batch(
                    portfolio_df,
                    fraud_input_df=portfolio_fraud_df,
                )
            except Exception as exc:
                st.error(f"Portfolio analysis failed: {exc}")
            else:
                st.session_state["scored_portfolio_df"] = batch_result["scored_portfolio"]
                st.session_state["validation_failures_df"] = batch_result["validation_failures"]
                st.session_state["portfolio_metrics"] = batch_result

    scored_portfolio_df = st.session_state["scored_portfolio_df"]
    validation_failures_df = st.session_state["validation_failures_df"]
    portfolio_metrics = st.session_state["portfolio_metrics"]
    if scored_portfolio_df is not None and portfolio_metrics is not None:
        summary = portfolio_metrics["summary_metrics"]

        st.subheader("Portfolio Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", summary["total_rows"])
        m2.metric("Success Rows", summary["success_rows"])
        m3.metric("Validation Failures", summary["validation_failed_rows"])
        m4.metric("Inference Failures", summary["inference_failed_rows"])

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Approve Count", summary["approve_count"])
        m6.metric("Review Count", summary["review_count"])
        m7.metric("Reject Count", summary["reject_count"])
        m8.metric("High Fraud Count", summary["high_fraud_count"])

        m9, m10 = st.columns(2)
        m9.metric("Approval Rate", f'{summary["approval_rate"]}%')
        m10.metric("Rejection Rate", f'{summary["rejection_rate"]}%')

        m11, m12 = st.columns(2)
        m11.metric(
            "Mean Credit Score",
            summary["mean_credit_score"] if summary["mean_credit_score"] is not None else "N/A",
        )
        m12.metric(
            "Mean Fraud Score",
            summary["mean_fraud_score"] if summary["mean_fraud_score"] is not None else "N/A",
        )

        if summary["validation_failed_rows"] > 0:
            st.warning(
                f"{summary['validation_failed_rows']} rows failed validation and were skipped before scoring."
            )
        if summary["inference_failed_rows"] > 0:
            st.warning(
                f"{summary['inference_failed_rows']} rows failed during model inference. Review 'scoring_error' in the output table."
            )

        st.divider()

        st.subheader("Approval / Rejection Counts")
        st.dataframe(
            pd.DataFrame(
                portfolio_metrics["approval_rejection_counts"].items(),
                columns=["Final Decision", "Count"],
            ),
            use_container_width=True,
        )

        st.subheader("Fraud Distribution")
        st.dataframe(
            pd.DataFrame(
                portfolio_metrics["fraud_distribution"].items(),
                columns=["Fraud Category", "Count"],
            ),
            use_container_width=True,
        )

        st.subheader("Credit Risk Distribution")
        st.dataframe(
            pd.DataFrame(
                portfolio_metrics["credit_risk_distribution"].items(),
                columns=["Credit Risk Category", "Count"],
            ),
            use_container_width=True,
        )

        st.subheader("Scored Portfolio Output")
        st.dataframe(scored_portfolio_df, use_container_width=True)
        st.download_button(
            "Download Scored CSV",
            data=convert_dataframe_to_csv(scored_portfolio_df),
            file_name="scored_portfolio.csv",
            mime="text/csv",
        )

        st.subheader("Validation Failures")
        if validation_failures_df is not None and not validation_failures_df.empty:
            st.dataframe(validation_failures_df, use_container_width=True)
        else:
            st.info("No validation failures were recorded.")

        st.download_button(
            "Download Validation Failures CSV",
            data=convert_dataframe_to_csv(
                validation_failures_df
                if validation_failures_df is not None
                else pd.DataFrame(columns=["source_row_index", "validation_status", "validation_error"])
            ),
            file_name="validation_failures.csv",
            mime="text/csv",
        )
