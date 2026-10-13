
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "churn_model.joblib"
DATA_FILE = ROOT / "data" / "generated" / "customer_360.csv"
OUTPUT_FILE = ROOT / "data" / "generated" / "churn_model_coefficients.csv"


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    excluded = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }

    X = df.drop(columns=list(excluded))

    # Extract the fitted preprocessing and classifier.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    explanations = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    explanations["absolute_coefficient"] = (
        explanations["coefficient"].abs()
    )

    explanations["direction"] = explanations[
        "coefficient"
    ].apply(
        lambda value: (
            "Higher churn score"
            if value > 0
            else "Lower churn score"
            if value < 0
            else "No linear effect"
        )
    )

    explanations = explanations.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\nTop 15 features by absolute coefficient:")
    print(
        explanations.head(15)[
            ["feature", "coefficient", "direction"]
        ].to_string(index=False)
    )

    # Export the complete explanation table.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved explanations to:", OUTPUT_FILE)

    # Example prediction for one customer.
    customer = X.iloc[[0]]
    probability = model.predict_proba(customer)[0, 1]

    print("\nExample customer:", df.iloc[0]["customer_id"])
    print(f"Predicted churn probability: {probability:.1%}")
    print(
        "Note: this is a model estimate on synthetic data, "
        "not a real customer's risk."
    )


if __name__ == "__main__":
    main()
