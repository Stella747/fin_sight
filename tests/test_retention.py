
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOMMENDATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "retention_recommendations.csv"
)


def load_recommendations():
    return pd.read_csv(RECOMMENDATIONS_FILE)


def test_recommendations_are_not_empty():
    recommendations = load_recommendations()

    assert len(recommendations) > 0


def test_each_customer_has_at_most_one_offer():
    recommendations = load_recommendations()

    assert "customer_id" in recommendations.columns
    assert recommendations["customer_id"].is_unique



def test_offer_costs_are_non_negative():
    recommendations = load_recommendations()

    assert "offer_cost_inr" in recommendations.columns
    assert (recommendations["offer_cost_inr"] >= 0).all()



def test_selected_offers_do_not_exceed_limit():
    recommendations = load_recommendations()

    assert len(recommendations) <= 500
