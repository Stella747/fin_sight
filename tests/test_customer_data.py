
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOMER_FILE = (
    PROJECT_ROOT / "data" / "generated" / "customer_360.csv"
)


def load_customers():
    return pd.read_csv(CUSTOMER_FILE)


def test_customer_dataset_is_not_empty():
    customers = load_customers()

    assert len(customers) > 0


def test_customer_ids_are_unique():
    customers = load_customers()

    assert "customer_id" in customers.columns
    assert customers["customer_id"].is_unique


def test_required_customer_columns_exist():
    customers = load_customers()

    required_columns = {
        "customer_id",
        "churned",
        "customer_segment",
    }

    assert required_columns.issubset(customers.columns)


def test_churn_target_contains_only_binary_values():
    customers = load_customers()

    assert customers["churned"].notna().all()
    assert set(customers["churned"].unique()).issubset({0, 1})
