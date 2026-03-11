from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=None):
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")

    return config


# Optional test block (recommended)
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully.\n")
    print(config)
