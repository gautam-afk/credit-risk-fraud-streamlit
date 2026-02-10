import pandas as pd
from pathlib import Path
from src.config.config_loader import load_config


def load_dataset(module_name):
    config = load_config()
    module_config = config[module_name]

    path = Path(module_config["dataset_path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    return df, module_config


# Test block
if __name__ == "__main__":
    df, cfg = load_dataset("credit_risk")
    print("Dataset shape:", df.shape)
    print("Numerical features:", cfg["numerical_features"])
