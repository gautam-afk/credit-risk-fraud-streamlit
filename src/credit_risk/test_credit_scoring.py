import pandas as pd
from src.preprocessing.dataset_loader import load_dataset
from src.credit_risk.credit_scoring import predict_credit_risk

# Load credit dataset
df, config = load_dataset("credit_risk")

# Take one sample applicant
sample = df.head(1)

# Run scoring
result = predict_credit_risk(sample)

print(result)
