
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the FinSight API"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    customer = {
        "age": 35,
        "tenure_years": 4,
        "annual_income_inr": 900000,
        "customer_segment": "Mass",
        "region": "South",
        "digital_preference": "High",
        "monthly_transactions": 30,
        "average_balance_inr": 50000,
        "credit_utilization": 0.35,
        "app_logins_last_30d": 12,
        "support_tickets_last_90d": 1,
        "days_since_last_transaction": 10,
        "salary_growth_rate": 0.05,
        "transaction_count": 120,
        "total_transaction_amount_inr": 240000,
        "average_transaction_amount_inr": 2000,
        "transaction_recency_days": 10,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_churn"] in [True, False]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["no_churn_probability"] <= 1

    total = (
        result["churn_probability"]
        + result["no_churn_probability"]
    )
    assert abs(total - 1) < 1e-5


def test_predict_rejects_invalid_age():
    customer = {
        "age": 10,
        "tenure_years": 4,
        "annual_income_inr": 900000,
        "customer_segment": "Mass",
        "region": "South",
        "digital_preference": "High",
        "monthly_transactions": 30,
        "average_balance_inr": 50000,
        "credit_utilization": 0.35,
        "app_logins_last_30d": 12,
        "support_tickets_last_90d": 1,
        "days_since_last_transaction": 10,
        "salary_growth_rate": 0.05,
        "transaction_count": 120,
        "total_transaction_amount_inr": 240000,
        "average_transaction_amount_inr": 2000,
        "transaction_recency_days": 10,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_missing_required_fields():
    response = client.post("/predict", json={"age": 35})

    assert response.status_code == 422


def test_predict_rejects_invalid_credit_utilization():
    customer = {
        "age": 35,
        "tenure_years": 4,
        "annual_income_inr": 900000,
        "customer_segment": "Mass",
        "region": "South",
        "digital_preference": "High",
        "monthly_transactions": 30,
        "average_balance_inr": 50000,
        "credit_utilization": 1.5,
        "app_logins_last_30d": 12,
        "support_tickets_last_90d": 1,
        "days_since_last_transaction": 10,
        "salary_growth_rate": 0.05,
        "transaction_count": 120,
        "total_transaction_amount_inr": 240000,
        "average_transaction_amount_inr": 2000,
        "transaction_recency_days": 10,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_negative_transaction_count():
    customer = {
        "age": 35,
        "tenure_years": 4,
        "annual_income_inr": 900000,
        "customer_segment": "Mass",
        "region": "South",
        "digital_preference": "High",
        "monthly_transactions": 30,
        "average_balance_inr": 50000,
        "credit_utilization": 0.35,
        "app_logins_last_30d": 12,
        "support_tickets_last_90d": 1,
        "days_since_last_transaction": 10,
        "salary_growth_rate": 0.05,
        "transaction_count": -1,
        "total_transaction_amount_inr": 240000,
        "average_transaction_amount_inr": 2000,
        "transaction_recency_days": 10,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_unknown_customer_segment():
    customer = {
        "age": 35,
        "tenure_years": 4,
        "annual_income_inr": 900000,
        "customer_segment": "Premum",
        "region": "South",
        "digital_preference": "High",
        "monthly_transactions": 30,
        "average_balance_inr": 50000,
        "credit_utilization": 0.35,
        "app_logins_last_30d": 12,
        "support_tickets_last_90d": 1,
        "days_since_last_transaction": 10,
        "salary_growth_rate": 0.05,
        "transaction_count": 120,
        "total_transaction_amount_inr": 240000,
        "average_transaction_amount_inr": 2000,
        "transaction_recency_days": 10,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
