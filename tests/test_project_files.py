
from pathlib import Path

# Find the root folder of the FinSight project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_customer_data_exists():
    """Check that the customer-level dataset exists."""
    file_path = PROJECT_ROOT / "data" / "generated" / "customer_360.csv"

    assert file_path.exists(), f"Missing file: {file_path}"


def test_retention_recommendations_exist():
    """Check that retention recommendations were generated."""
    file_path = (
        PROJECT_ROOT
        / "data"
        / "generated"
        / "retention_recommendations.csv"
    )

    assert file_path.exists(), f"Missing file: {file_path}"


def test_churn_model_exists():
    """Check that the trained churn model was saved."""
    file_path = PROJECT_ROOT / "models" / "churn_model.joblib"

    assert file_path.exists(), f"Missing file: {file_path}"
