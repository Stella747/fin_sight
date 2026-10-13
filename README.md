# FinSight — AI-Powered Financial Decision Intelligence

FinSight is a local-first financial analytics and decision-support prototype that combines customer analytics, churn prediction, explainable AI, retention-offer optimization, experimentation, and an interactive dashboard.

**Project status:** Working prototype  
**Data:** Synthetic customer data only  
**Cost:** Designed to run locally using free, open-source tools

## Features

- **Customer analytics:** Customer-level data preparation, transaction aggregation, and segment analysis using pandas and DuckDB.
- **Churn prediction:** A scikit-learn pipeline that estimates customer churn probability.
- **Explainable AI:** Model coefficients and SHAP visualizations to help interpret model behavior.
- **Retention optimization:** OR-Tools selects retention offers under configurable budget and offer-count constraints.
- **Experiment analysis:** A simulated A/B experiment calculates conversion rates, absolute lift, confidence intervals, and statistical significance.
- **Interactive dashboard:** Streamlit and Plotly present analytics and recommendations.
- **Prediction API:** FastAPI provides health-check and customer churn-prediction endpoints.
- **Automated tests:** pytest checks project files, data integrity, model outputs, business rules, and API validation.

## Technology stack

Python, pandas, NumPy, DuckDB, scikit-learn, SHAP, OR-Tools, Streamlit, Plotly, FastAPI, Uvicorn, pytest, and HTTPX.

## Local setup

Requirements: Python and Git.

Create and activate a virtual environment in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies used by the scripts, dashboard, API, and tests:

```powershell
python -m pip install pandas numpy openpyxl duckdb scikit-learn shap matplotlib ortools scipy statsmodels streamlit plotly fastapi uvicorn httpx pytest joblib
```

## Generate the data and model

Run the scripts from the project root, in this order:

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

These scripts create the synthetic datasets, model, explanations, recommendations, and experiment results used by the dashboard and API.

## Run the dashboard

```powershell
python -m streamlit run app.py
```

## Run the prediction API

```powershell
python -m uvicorn api:app --reload
```

Local endpoints:

- `GET /` — API welcome message
- `GET /health` — model availability
- `POST /predict` — predict churn from customer features
- `/docs` — interactive API documentation

The prediction request requires the 17 feature fields defined by the API schema.

## Run automated tests

```powershell
python -m pytest -v
```

The current suite contains 27 tests. Tests should be rerun after changes to the data pipeline, model, optimization logic, or API.

## Important limitations

- All customer records and experiment results are synthetic.
- Retention success rates, customer values, and channel costs are assumptions for demonstrating optimization—not validated business estimates.
- Churn probabilities come from a prototype model trained on synthetic data. They should not be interpreted as calibrated real-world risk scores.
- A statistically significant result in the simulated experiment does not establish that a real campaign will improve conversion.
- The project is intended for learning and portfolio demonstration, not production financial decision-making.

## Future improvements

- Improve temporal validation and model evaluation.
- Add model calibration and decision-threshold analysis.
- Add stronger automated tests and API error handling.
- Connect the dashboard to the prediction API.
- Document reproducible experiments and deployment options.

## License

Add a license before distributing the project as open source. Until a license is selected, reuse permissions are not explicitly granted.