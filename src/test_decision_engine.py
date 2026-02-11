from src.preprocessing.dataset_loader import load_dataset
from src.decision_engine import generate_final_decision

df, config = load_dataset("credit_risk")

sample = df.head(1)

result = generate_final_decision(sample)

print(result)
