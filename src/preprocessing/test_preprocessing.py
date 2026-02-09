from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import prepare_data

df, config = load_dataset("fraud_detection")

X, y, preprocessor = prepare_data(
    df,
    config["target_column"],
    config["numerical_features"],
    config["categorical_features"]
)

print("X shape:", X.shape)
print("y length:", len(y))
