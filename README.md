# FinSight

FinSight is a local-first financial analytics prototype that combines synthetic customer data, churn prediction, retention-offer optimization, experiment analysis, and an interactive dashboard. A FastAPI service exposes churn predictions for customer feature records.

> **Portfolio demonstration only:** Customer data and experiment outcomes are synthetic. Model scores, offer values, and campaign results are illustrative and are not validated for real financial decisions.

## Features

- **Customer analytics:** Generate source tables, build a customer-level dataset, aggregate transactions, and summarize customer segments.
- **Churn modeling:** Train a scikit-learn classification pipeline and inspect model coefficients and SHAP explanations.
- **Retention optimization:** Select offers with OR-Tools subject to budget and offer-count constraints.
- **Experiment analysis:** Simulate a control/treatment experiment and calculate conversion rates, lift, confidence intervals, and a p-value.
- **Interactive dashboard:** Explore results across five Streamlit pages: Executive Overview, Customer Intelligence, Retention Optimization, Experimentation, and Model Explainability.
- **Prediction API:** Validate customer features and return churn probabilities with FastAPI and Pydantic.

## Architecture

- `scripts/generate_data.py` creates a synthetic Excel workbook with customer, product-usage, and transaction tables.
- `scripts/build_customer_360.py` validates and combines the source tables into a customer-level CSV.
- `scripts/run_analytics.py`, `scripts/train_churn_model.py`, `scripts/explain_churn_model.py`, `scripts/explain_with_shap.py`, `scripts/optimize_retention.py`, and `scripts/run_experiment.py` create the analysis outputs, model, explanations, recommendations, and simulated experiment results.
- `app.py` reads generated data and model artifacts to power the Streamlit dashboard.
- `api.py` loads the trained model and serves validated prediction requests.
- `tests/` covers project artifacts, data integrity, model outputs, retention rules, experiment calculations, and API validation.

Generated datasets, model files, and SHAP images are excluded from version control. Generate these local artifacts before starting the dashboard, making API predictions, or running the full test suite.

## Tech stack

Python, pandas, NumPy, DuckDB, scikit-learn, SHAP, OR-Tools, Streamlit, Plotly, FastAPI, Pydantic, Uvicorn, pytest, and HTTPX.

## Quick Start

Requirements: Python and Git.

Create a virtual environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

From the project root, generate the data, model, explanations, recommendations, and experiment results in order:

```powershell
python scripts/generate_data.py
python scripts/build_customer_360.py
python scripts/run_analytics.py
python scripts/train_churn_model.py
python scripts/explain_churn_model.py
python scripts/explain_with_shap.py
python scripts/optimize_retention.py
python scripts/run_experiment.py
```

### Run the dashboard

```powershell
python -m streamlit run app.py
```

### Run the prediction API

```powershell
python -m uvicorn api:app --reload
```

API routes:

- `GET /` — API welcome message
- `GET /health` — model availability
- `POST /predict` — predict churn from 17 validated customer feature fields
- `/docs` — interactive API documentation

### Run automated tests

After generating the local artifacts, run:

```powershell
python -m pytest -v
```

The current suite contains 27 tests.

## Limitations

- Customer records and experiment outcomes are synthetic.
- The churn model learns patterns from a generated target and should not be treated as a calibrated real-world risk model.
- Retention success rates, customer values, and channel costs are illustrative assumptions, not measured business estimates.
- The experiment generator uses a preselected treatment effect; its results do not establish that a real campaign will improve conversion.
- This project is for learning and portfolio demonstration, not production financial decision-making.


