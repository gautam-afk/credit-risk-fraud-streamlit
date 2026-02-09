import yaml


def load_config(path="src/config/config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Optional test block (recommended)
if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully.\n")
    print(config)
