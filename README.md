# Credit Risk And Fraud Detection

Modular ML system for:

- credit risk scoring on German Credit applications
- fraud scoring on transaction inputs
- combined decisioning where fraud can override credit
- Streamlit-based manual scoring and portfolio analytics

## Architecture

```text
app/
  streamlit_app.py
app.py
src/
  cli/
    batch_scoring.py
  common/
    logging_utils.py
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
    dataset_loader.py
    data_inspection.py
    preprocessor.py
    test_preprocessing.py
  decision_engine.py
  portfolio_analytics.py
  test_decision_engine.py
  test_portfolio_analytics.py
requirements.txt
```

## Module Overview

### Credit module

- Dataset: `data/raw/german_credit.csv`
- Features:
  - numerical: `age`, `credit_amount`, `month_duration`
  - categorical: `housing`, `years_employment`, `purpose`
- Target:
  - `target`
  - encoded as `good -> 0`, `bad -> 1`
- Outputs:
  - `credit_probability`
  - `credit_score`
  - `credit_category`
  - `credit_decision`
  - `credit_latency_ms`

### Fraud module

- Dataset: `data/raw/fraud_dataset.csv`
- Features:
  - numerical: `Amount`, `Time`
- Target:
  - `Class`
  - numeric `0/1`
- Outputs:
  - `fraud_probability`
  - `fraud_score`
  - `fraud_category`
  - `fraud_decision`
  - `fraud_latency_ms`

### Decision engine

- Runs credit scoring on the credit feature space
- Runs fraud scoring only when fraud columns are present
- Applies override logic:
  - `High Fraud Risk` -> `Reject`
  - `Medium Fraud Risk` -> `Review`
  - otherwise -> use `credit_decision`

## Training

Train from the repository root:

```bash
python -m src.credit_risk.train_credit_model
python -m src.fraud_detection.train_fraud_model
```

Saved artifacts:

- `models/credit_model.pkl`
- `models/fraud_model.pkl`

## Run The UI

```bash
streamlit run app/streamlit_app.py
```

## Deploy Online

### 1) Push to GitHub

From the repository root:

```bash
git add .
git commit -m "Fix scoring compatibility and deployment docs"
git push origin main
```

### 2) Deploy on Streamlit Community Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select repository: `gautam-afk/credit-risk-fraud-streamlit`.
4. Set branch: `main`.
5. Set main file path: `app/streamlit_app.py`.
6. Click **Deploy**.

After deployment, Streamlit provides a public app URL that stays linked to this GitHub repository.

The applicant form collects:

- credit fields:
  - `age`
  - `credit_amount`
  - `month_duration`
  - `housing`
  - `years_employment`
  - `purpose`
- fraud fields:
  - `Amount`
  - `Time`

The portfolio tab accepts a CSV and summarizes:

- average credit probability and score
- fraud distribution
- approval / review / rejection counts
- latency-aware portfolio metrics

## Run Batch Scoring From CLI

Score a credit or combined CSV from the terminal:

```bash
python -m src.cli.batch_scoring input.csv outputs/
```

With a separate fraud CSV aligned row-for-row:

```bash
python -m src.cli.batch_scoring credit.csv outputs/ --fraud-csv fraud.csv --chunk-size 2000
```

CLI artifacts:

- `scored_portfolio.csv`
- `validation_failures.csv`
- `portfolio_metrics.json`

CLI behavior:

- processes large files in chunks
- continues when individual rows fail validation or inference
- exits with code `1` if every row fails
- exits with code `2` on fatal runtime errors

## Batch Output Contract

The enriched scored CSV preserves the original input columns and adds:

- `credit_probability`
- `credit_score`
- `credit_category`
- `fraud_probability`
- `fraud_score`
- `fraud_category`
- `final_decision`
- `scoring_status`
- `scoring_error`

`scoring_status` values:

- `scored`
- `validation_failed`
- `inference_failed`

The validation report CSV contains:

- original row payload
- `source_row_index`
- `validation_status`
- `validation_error`

## Batch Processing Notes

The portfolio engine:

- scores rows individually through the decision engine
- separates `validation_failed` rows from `inference_failed` rows
- writes validation failures to a dedicated report
- maintains portfolio metrics incrementally during chunked processing

## Testing

Run smoke tests from the repository root:

```bash
python -m src.credit_risk.test_credit_scoring
python -m src.fraud_detection.test_fraud_scoring
python -m src.test_decision_engine
python -m src.test_portfolio_analytics
python -m src.preprocessing.test_preprocessing
```

## Design Notes

- Model bundles are resolved relative to project root and cached lazily.
- Scoring functions enforce single-row inference and align feature order before preprocessing.
- Extra columns are ignored safely.
- Missing required columns raise clean exceptions.
- Fraud scoring does not derive features from credit inputs.
