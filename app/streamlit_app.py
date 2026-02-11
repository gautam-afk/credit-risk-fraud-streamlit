import streamlit as st
import pandas as pd
from src.decision_engine import generate_final_decision

st.set_page_config(page_title="Credit Risk System")

st.title("Credit Risk and Fraud Detection System")

st.header("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)

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

    st.subheader("Final Decision")
    st.write(result["final_decision"])

    st.subheader("Credit Details")
    st.write(result["credit"])

    st.subheader("Fraud Details")
    st.write(result["fraud"])
