from pathlib import Path

import pandas as pd


def inspect_dataset(name, dataset_path, target_column):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"Loading {name} Dataset...\n")
    print(f"Shape of {name} Dataset:")
    print(df.shape)
    print("\nColumns:")
    print(list(df.columns))
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nNull Values:")
    print(df.isnull().sum())
    print(f"\nTarget Distribution ({name}):")
    print(df[target_column].value_counts())

    return df


def main():
    inspect_dataset("Credit Risk", "data/raw/german_credit.csv", "target")
    print("\n\n")
    inspect_dataset("Fraud", "data/raw/fraud_dataset.csv", "Class")


if __name__ == "__main__":
    main()
