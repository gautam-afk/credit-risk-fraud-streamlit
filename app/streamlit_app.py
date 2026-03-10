import streamlit as st
import pandas as pd
from src.decision_engine import generate_final_decision

st.set_page_config(page_title="Credit Risk System")

st.title("Credit Risk and Fraud Detection System")

st.header("Enter Applicant Details")

# -----------------------------
# Input Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)

with col2:
    month_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)
    housing = st.selectbox("Housing", ["own", "rent", "free"])

years_employment = st.selectbox(
    "Years of Employment",
    ["unemployed", "<1 year", "1-4 years", "4-7 years", ">=7 years"]
)

purpose = st.selectbox(
    "Loan Purpose",
    ["car", "furniture", "radio/TV", "education", "business", "other"]
)

st.divider()

# -----------------------------
# Run Decision Engine
# -----------------------------
if st.button("Evaluate Applicant"):

    input_data = pd.DataFrame([{
        "age": age,
        "credit_amount": credit_amount,
        "month_duration": month_duration,
        "housing": housing,
        "years_employment": years_employment,
        "purpose": purpose
    }])

    result = generate_final_decision(input_data)

    # -----------------------------
    # Final Decision
    # -----------------------------
    st.subheader("Final Decision")
    decision = result["final_decision"]

    if decision == "Approve":
        st.success(f"Decision: {decision}")
    elif decision == "Review":
        st.warning(f"Decision: {decision}")
    else:
        st.error(f"Decision: {decision}")

    st.divider()

    # -----------------------------
    # Credit Risk Section
    # -----------------------------
    st.subheader("Credit Risk Analysis")
    credit = result["credit"]
    c1, c2, c3 = st.columns(3)

    c1.metric("Probability of Default", credit["probability_default"])
    c2.metric("Risk Score", credit["risk_score"])
    c3.metric("Risk Category", credit["risk_category"])

    st.divider()

    # -----------------------------
    # Fraud Risk Section
    # -----------------------------
    st.subheader("Fraud Risk Analysis")
    fraud = result["fraud"]
    f1, f2, f3 = st.columns(3)

    f1.metric("Fraud Probability", fraud["probability_fraud"])
    f2.metric("Fraud Score", fraud["fraud_score"])
    f3.metric("Fraud Category", fraud["fraud_category"])
