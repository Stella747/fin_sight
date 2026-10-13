
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "generated" / "customer_360.csv"
MODEL_FILE = ROOT / "models" / "churn_model.joblib"
OUTPUT_DIR = ROOT / "reports" / "shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    excluded = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }

    X = df.drop(columns=list(excluded))

    # Explain a small sample to keep computation manageable.
    sample = X.sample(
        n=min(100, len(X)),
        random_state=42,
    )

    # Recover the preprocessing and fitted classifier.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # SHAP will explain the classifier after preprocessing.
    transformed = preprocessor.transform(sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=sample.index,
    )

    # LinearExplainer is suitable for Logistic Regression.
    explainer = shap.LinearExplainer(
        classifier,
        transformed_df,
    )

    shap_values = explainer(transformed_df)

    # 1. Global feature importance.
    plt.figure()
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "global_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Global distribution of SHAP contributions.
    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "global_shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Explain the first customer in the sample.
    plt.figure()
    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "individual_customer_explanation.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Compare the explained model output with its prediction.
    # For Logistic Regression, SHAP explains the raw linear
    # output (log-odds), not probability directly.
    customer = sample.iloc[[0]]
    probability = model.predict_proba(customer)[0, 1]

    print("SHAP analysis completed.")
    print("Customer index:", sample.index[0])
    print(f"Predicted churn probability: {probability:.1%}")
    print("Plots saved in:", OUTPUT_DIR)
    print("1. global_feature_importance.png")
    print("2. global_shap_summary.png")
    print("3. individual_customer_explanation.png")


if __name__ == "__main__":
    main()
