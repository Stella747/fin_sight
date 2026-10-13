
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


# Locate the trained model relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

app = FastAPI(
    title="FinSight API",
    description="Customer churn prediction service",
    version="1.0.0",
)

FEATURE_COLUMNS = [
    "age",
    "tenure_years",
    "annual_income_inr",
    "customer_segment",
    "region",
    "digital_preference",
    "monthly_transactions",
    "average_balance_inr",
    "credit_utilization",
    "app_logins_last_30d",
    "support_tickets_last_90d",
    "days_since_last_transaction",
    "salary_growth_rate",
    "transaction_count",
    "total_transaction_amount_inr",
    "average_transaction_amount_inr",
    "transaction_recency_days",
]


# Load the fitted pipeline once when the API starts.
try:
    model = joblib.load(MODEL_PATH)
    model_load_error = None
except Exception as exc:
    model = None
    model_load_error = str(exc)


class CustomerFeatures(BaseModel):
    age: float = Field(ge=18, le=120)
    tenure_years: float = Field(ge=0)
    annual_income_inr: float = Field(ge=0)
    customer_segment: Literal["Affluent", "Mass", "Premium"]
    region: Literal["Central", "East", "North", "South", "West"]
    digital_preference: Literal["High", "Low", "Medium"]
    monthly_transactions: float = Field(ge=0)
    average_balance_inr: float
    credit_utilization: float = Field(ge=0, le=1)
    app_logins_last_30d: float = Field(ge=0)
    support_tickets_last_90d: float = Field(ge=0)
    days_since_last_transaction: float = Field(ge=0)
    salary_growth_rate: float
    transaction_count: float = Field(ge=0)
    total_transaction_amount_inr: float = Field(ge=0)
    average_transaction_amount_inr: float = Field(ge=0)
    transaction_recency_days: float = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Welcome to the FinSight API"}


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_error": model_load_error,
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Churn model is unavailable.",
        )

    # Preserve the exact feature names and order used in training.
    customer_df = pd.DataFrame(
        [customer.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    probabilities = model.predict_proba(customer_df)[0]
    class_probabilities = dict(zip(model.classes_, probabilities))

    churn_probability = float(class_probabilities.get(1, 0.0))
    no_churn_probability = float(class_probabilities.get(0, 0.0))
    predicted_class = int(model.predict(customer_df)[0])

    return {
        "customer_id": "not_provided",
        "predicted_churn": predicted_class == 1,
        "churn_probability": round(churn_probability, 6),
        "no_churn_probability": round(no_churn_probability, 6),
        "model_classes": [int(value) for value in model.classes_],
        "note": "Prediction from a model trained on synthetic data.",
    }
