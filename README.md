# Credit Risk and Fraud Detection

Config-driven ML system for:
- credit risk scoring (German Credit)
- fraud risk scoring (credit card fraud features)
- unified final decisioning with fraud override
- interactive Streamlit demo

## Project Structure

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
  preprocessing/
    data_inspection.py
    dataset_loader.py
    preprocessor.py
    test_preprocessing.py
  credit_risk/
    train_credit_model.py
    credit_scoring.py
    test_credit_scoring.py
  fraud_detection/
    train_fraud_model.py
    fraud_scoring.py
    test_fraud_scoring.py
  decision_engine.py
  test_decision_engine.py
app.py
requirements.txt
```

## Datasets

Paths are configured in `src/config/config.yaml`:
- `credit_risk.dataset_path`: `data/raw/german_credit.csv`
- `fraud_detection.dataset_path`: `data/raw/fraud_dataset.csv`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Train Models

Run from project root:

```bash
python -m src.credit_risk.train_credit_model
python -m src.fraud_detection.train_fraud_model
```

Artifacts are saved to:
- `models/credit_model.pkl`
- `models/fraud_model.pkl`

## Smoke Tests

```bash
python -m src.credit_risk.test_credit_scoring
python -m src.fraud_detection.test_fraud_scoring
python -m src.test_decision_engine
python -m src.preprocessing.test_preprocessing
```

## Run App

```bash
python -m streamlit run app/streamlit_app.py
```

Optional helper:

```bash
python app.py
```

## Decision Rules

Credit model output:
- probability of default
- risk score (`(1 - prob_default) * 100`)
- risk category (`Low`, `Medium`, `High`)
- credit decision (`Approve`, `Review`, `Reject`)

Fraud model output:
- probability of fraud
- fraud score (`prob_fraud * 100`)
- fraud category (`Low Fraud Risk`, `Medium Fraud Risk`, `High Fraud Risk`)
- fraud decision (`Clear`, `Review`, `Reject`)

Final decision in `src/decision_engine.py`:
- `High Fraud Risk` -> `Reject`
- `Medium Fraud Risk` -> `Review`
- otherwise -> credit decision

## Notes

- Train models before running scoring modules or Streamlit UI.
- Fraud scoring accepts either `Amount`/`Time` or mapped fields `credit_amount`/`month_duration`.
- For fraud, monitor confusion matrix and class-wise precision/recall/F1, not accuracy alone.
