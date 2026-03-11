import pandas as pd
from pathlib import Path
from src.config.config_loader import load_config


def load_dataset(module_name):
    config = load_config()
    if module_name not in config:
        available_modules = ", ".join(sorted(config.keys())) or "<none>"
        raise KeyError(
            f"Module '{module_name}' not found in config. Available: {available_modules}"
        )

    module_config = config[module_name]
    dataset_path = module_config.get("dataset_path")
    if not dataset_path:
        raise ValueError(f"'dataset_path' is missing for module '{module_name}' in config.")

    path = Path(dataset_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    return df, module_config


# Test block
if __name__ == "__main__":
    df, cfg = load_dataset("credit_risk")
    print("Dataset shape:", df.shape)
    print("Numerical features:", cfg["numerical_features"])
