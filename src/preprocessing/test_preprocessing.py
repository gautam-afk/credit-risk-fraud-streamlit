from src.preprocessing.dataset_loader import load_dataset
from src.preprocessing.preprocessor import prepare_data

def test_prepare_data_smoke():
    df, config = load_dataset("fraud_detection")

    X, y, _ = prepare_data(
        df,
        config["target_column"],
        config["numerical_features"],
        config["categorical_features"]
    )

    assert X.shape[0] == len(y)


if __name__ == "__main__":
    test_prepare_data_smoke()
