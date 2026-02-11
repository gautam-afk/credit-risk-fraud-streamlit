from src.preprocessing.dataset_loader import load_dataset
from src.fraud_detection.fraud_scoring import predict_fraud_risk

df, config = load_dataset("fraud_detection")

sample = df.head(1)

result = predict_fraud_risk(sample)

print(result)
