
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"
CUSTOMER_FILE = (
    PROJECT_ROOT / "data" / "generated" / "customer_360.csv"
)


def test_churn_model_loads():
    model = joblib.load(MODEL_FILE)

    assert model is not None


def test_model_predicts_churn_probabilities():
    model = joblib.load(MODEL_FILE)
    customers = pd.read_csv(CUSTOMER_FILE)

    # Match the feature selection used by the training script.
    excluded_columns = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }
    feature_columns = [
        column
        for column in customers.columns
        if column not in excluded_columns
    ]

    sample = customers[feature_columns].head(10)
    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (10, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_model_probabilities_sum_to_one():
    model = joblib.load(MODEL_FILE)
    customers = pd.read_csv(CUSTOMER_FILE)

    excluded_columns = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }
    feature_columns = [
        column
        for column in customers.columns
        if column not in excluded_columns
    ]

    sample = customers[feature_columns].head(10)
    probabilities = model.predict_proba(sample)

    assert (abs(probabilities.sum(axis=1) - 1) < 1e-6).all()
