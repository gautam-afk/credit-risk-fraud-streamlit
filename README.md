# Credit Risk and Fraud Detection

This project combines two scoring pipelines into one decision engine:

- credit risk scoring on the German Credit dataset
- fraud risk scoring on transaction-style features
- final decisioning where fraud risk can override credit approval
- a small Streamlit UI for manual evaluation

## Repository Layout

```text
app/
  streamlit_app.py
src/
  common/
    model_utils.py
    schemas.py
    validation.py
  config/
    config.yaml
    config_loader.py
  credit_risk/
    credit_scoring.py
    test_credit_scoring.py
    train_credit_model.py
  fraud_detection/
    fraud_scoring.py
    test_fraud_scoring.py
    train_fraud_model.py
  preprocessing/
    data_inspection.py
    dataset_loader.py
    preprocessor.py
    test_preprocessing.py
  decision_engine.py
  test_decision_engine.py
app.py
requirements.txt
```

## How It Works

### Credit risk pipeline

Configured in `src/config/config.yaml` with:

- dataset: `data/raw/german_credit.csv`
- target column: `target`
- numerical features: `age`, `credit_amount`, `month_duration`
- categorical features: `housing`, `years_employment`, `purpose`

Training:

- maps target labels `good -> 1` and `bad -> 0`
- imputes missing values
- standardizes numeric features
- one-hot encodes categorical features
- trains a `LogisticRegression` model

Scoring output:

- `probability_default`
- `risk_score = (1 - probability_default) * 100`
- `risk_category` in `Low Risk`, `Medium Risk`, `High Risk`
- `decision` in `Approve`, `Review`, `Reject`

### Fraud risk pipeline

Configured in `src/config/config.yaml` with:

- dataset: `data/raw/fraud_dataset.csv`
- target column: `Class`
- numerical features: `Amount`, `Time`

Training:

- imputes and scales numerical inputs
- trains a class-weighted `LogisticRegression`
- prints confusion matrix and classification report

Scoring output:

- `probability_fraud`
- `fraud_score = probability_fraud * 100`
- `fraud_category` in `Low Fraud Risk`, `Medium Fraud Risk`, `High Fraud Risk`
- `fraud_decision` in `Clear`, `Review`, `Reject`

The fraud scorer accepts either:

- direct fraud fields: `Amount`, `Time`
- mapped application fields: `credit_amount`, `month_duration`

### Final decision engine

`src/decision_engine.py` runs both models and applies fraud override logic:

- `High Fraud Risk` -> `Reject`
- `Medium Fraud Risk` -> `Review`
- otherwise -> credit decision

## Setup

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Dependencies:

- `streamlit`
- `joblib`
- `pandas`
- `scikit-learn`
- `PyYAML`

## Data Requirements

Expected dataset paths are defined in `src/config/config.yaml`:

- `credit_risk.dataset_path: data/raw/german_credit.csv`
- `fraud_detection.dataset_path: data/raw/fraud_dataset.csv`

`src/preprocessing/dataset_loader.py` resolves relative paths from the repository root and raises an error if a file is missing.

## Train Models

Run from the repository root:

```bash
python -m src.credit_risk.train_credit_model
python -m src.fraud_detection.train_fraud_model
```

Saved artifacts:

- `models/credit_model.pkl`
- `models/fraud_model.pkl`

Train the models before running scoring code, tests, or the Streamlit app.

## Run Tests

Smoke tests available in the repo:

```bash
python -m src.credit_risk.test_credit_scoring
python -m src.fraud_detection.test_fraud_scoring
python -m src.test_decision_engine
python -m src.preprocessing.test_preprocessing
```

## Run the App

Launch the Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

`app.py` is only a small helper that prints the Streamlit command.

## Streamlit Inputs

The UI collects:

- `age`
- `credit_amount`
- `month_duration`
- `housing`
- `years_employment`
- `purpose`

It then displays:

- final decision
- credit probability, score, and category
- fraud probability, score, and category

## Notes

- Keep feature names aligned with `src/config/config.yaml`.
- Credit scoring validates required columns against the saved preprocessor schema.
- Fraud evaluation should be judged with class-wise metrics and confusion matrix, not raw accuracy alone.
