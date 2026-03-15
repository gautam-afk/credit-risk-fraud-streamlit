import logging
from pathlib import Path

import pandas as pd
from src.common.logging_utils import get_logger


logger = get_logger(__name__)


def inspect_dataset(name, dataset_path, target_column):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    df = pd.read_csv(path)

    logger.info("Loading %s dataset from %s", name, path)
    logger.info("%s dataset shape: %s", name, df.shape)
    logger.info("%s dataset columns: %s", name, list(df.columns))
    logger.info("%s dataset preview:\n%s", name, df.head())
    logger.info("%s dataset null values:\n%s", name, df.isnull().sum())
    logger.info("%s target distribution:\n%s", name, df[target_column].value_counts())

    return df


def main():
    inspect_dataset("Credit Risk", "data/raw/german_credit.csv", "target")
    inspect_dataset("Fraud", "data/raw/fraud_dataset.csv", "Class")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
