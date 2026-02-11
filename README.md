# Credit Risk and Fraud Detection

A config-driven ML system that trains and serves:
- Credit risk model (German Credit dataset)
- Fraud risk model (Credit Card Fraud dataset)
- Unified decision engine (fraud override over credit)
- Streamlit UI for end-to-end evaluation

## Project Structure

```text
app/
  streamlit_app.py
src/
  config/
    config.yaml
    config_loader.py
  preprocessing/
    dataset_loader.py
    preprocessor.py
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
```

## Datasets

Configured in `src/config/config.yaml`:
- `credit_risk` -> `data/raw/german_credit.csv`
- `fraud_detection` -> `data/raw/fraud_dataset.csv`

## Installation

From project root:

```bash
python -m pip install -r requirements.txt
```

## Train Models

Use module mode for files inside `src/`:

```bash
python -m src.credit_risk.train_credit_model
python -m src.fraud_detection.train_fraud_model
```

Expected outputs include evaluation metrics and success messages.

## Run Scoring Tests

```bash
python -m src.credit_risk.test_credit_scoring
python -m src.fraud_detection.test_fraud_scoring
python -m src.test_decision_engine
```

`src.test_decision_engine` verifies unified output with:
- `credit` block
- `fraud` block
- `final_decision`

## Run Streamlit App

```bash
python -m streamlit run app/streamlit_app.py
```

If `streamlit` is available on PATH, this also works:

```bash
streamlit run app/streamlit_app.py
```

## Decision Logic

`src/decision_engine.py` applies:
- Credit scoring decision from `src/credit_risk/credit_scoring.py`
- Fraud scoring decision from `src/fraud_detection/fraud_scoring.py`
- Final override rule:
  - High fraud risk -> `Reject`
  - Medium fraud risk -> `Review`
  - Low fraud risk -> credit decision

## Notes

- Fraud data is imbalanced; accuracy alone is not enough.
- Prioritize confusion matrix, precision/recall/F1 for fraud class.
- Trained artifacts under `models/` are ignored by git (`.gitignore`).
