
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = (
    PROJECT_ROOT / "data" / "generated" / "experiment_results.csv"
)


def load_results():
    return pd.read_csv(RESULTS_FILE)


def test_experiment_results_exist():
    results = load_results()

    assert not results.empty


def test_both_experiment_groups_have_customers():
    results = load_results()

    assert (results["control_customers"] > 0).all()
    assert (results["treatment_customers"] > 0).all()


def test_conversion_rates_are_between_zero_and_one():
    results = load_results()

    assert results["control_rate"].between(0, 1).all()
    assert results["treatment_rate"].between(0, 1).all()


def test_p_value_is_valid():
    results = load_results()

    assert results["p_value"].between(0, 1).all()


def test_absolute_lift_matches_conversion_rates():
    results = load_results()

    calculated_lift = (
        results["treatment_rate"] - results["control_rate"]
    )

    assert (
        (calculated_lift - results["absolute_lift"]).abs() < 1e-9
    ).all()
