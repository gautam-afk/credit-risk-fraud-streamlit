import pandas as pd

# ---------------------------------
# CREDIT RISK DATASET
# ---------------------------------

print("Loading Credit Risk Dataset...\n")

credit_df = pd.read_csv("data/raw/german_credit.csv")

print("Shape of Credit Risk Dataset:")
print(credit_df.shape)

print("\nColumns:")
print(list(credit_df.columns))

print("\nFirst 5 Rows:")
print(credit_df.head())

print("\nNull Values:")
print(credit_df.isnull().sum())

print("\nTarget Distribution (Credit Risk):")
print(credit_df["target"].value_counts())


# ---------------------------------
# FRAUD DATASET
# ---------------------------------

print("\n\nLoading Fraud Dataset...\n")

fraud_df = pd.read_csv("data/raw/fraud_dataset.csv")

print("Shape of Fraud Dataset:")
print(fraud_df.shape)

print("\nColumns:")
print(list(fraud_df.columns))

print("\nFirst 5 Rows:")
print(fraud_df.head())

print("\nNull Values:")
print(fraud_df.isnull().sum())

print("\nTarget Distribution (Fraud):")
print(fraud_df["Class"].value_counts())
