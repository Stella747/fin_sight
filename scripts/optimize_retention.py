
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "generated" / "customer_360.csv"
MODEL_FILE = ROOT / "models" / "churn_model.joblib"
OUTPUT_FILE = (
    ROOT / "data" / "generated"
    / "retention_recommendations.csv"
)

# Synthetic business assumptions for demonstration only.
BUDGET_INR = 100_000
MAX_OFFERS = 500
MAX_OFFERS_PER_CUSTOMER = 1

# Illustrative estimated retention success if contacted.
# These values are NOT learned from real experiments.
OFFER_SUCCESS = {
    "Mobile App": 0.12,
    "Email": 0.08,
    "SMS": 0.06,
}

CHANNEL_COST = {
    "Mobile App": 20,
    "Email": 8,
    "SMS": 12,
}

# Simulated value of retaining a customer, based on segment.
CUSTOMER_VALUE = {
    "Mass": 5_000,
    "Affluent": 15_000,
    "Premium": 25_000,
}


def main():
    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    excluded = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }
    X = df.drop(columns=list(excluded))

    # Score all customers.
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # Generate one candidate action per customer per channel.
    candidates = []

    for _, customer in df.iterrows():
        customer_id = customer["customer_id"]
        churn_probability = float(
            customer["churn_probability"]
        )

        # This is a simple simulated value proxy.
        customer_value = CUSTOMER_VALUE.get(
            customer["customer_segment"], 5_000
        )

        for channel in CHANNEL_COST:
            cost = CHANNEL_COST[channel]
            success_rate = OFFER_SUCCESS[channel]

            # Expected retained value under the assumed policy:
            # P(churn) * P(retained by offer) * customer value.
            expected_benefit = (
                churn_probability
                * success_rate
                * customer_value
            )

            expected_net_value = expected_benefit - cost

            # CP-SAT uses integer objective coefficients.
            candidates.append({
                "customer_id": customer_id,
                "channel": channel,
                "churn_probability": churn_probability,
                "assumed_retention_success": success_rate,
                "customer_value_inr": customer_value,
                "offer_cost_inr": cost,
                "expected_benefit_inr": expected_benefit,
                "expected_net_value_inr": expected_net_value,
            })

    options = pd.DataFrame(candidates)

    # Do not select offers with negative estimated net value.
    options = options[
        options["expected_net_value_inr"] > 0
    ].reset_index(drop=True)

    if options.empty:
        raise RuntimeError(
            "No positive-value candidate offers found."
        )

    # Integer rupees preserve objective ordering for this demo.
    options["objective_value"] = (
        options["expected_net_value_inr"] * 100
    ).round().astype(int)

    options["cost_integer"] = (
        options["offer_cost_inr"] * 100
    ).astype(int)

    # Optimization model.
    optimization = cp_model.CpModel()
    selected = [
        optimization.NewBoolVar(f"offer_{i}")
        for i in range(len(options))
    ]

    # Budget constraint.
    optimization.Add(
        sum(
            int(options.loc[i, "cost_integer"]) * selected[i]
            for i in range(len(options))
        )
        <= BUDGET_INR * 100
    )

    # Maximum number of offers.
    optimization.Add(
        sum(selected) <= MAX_OFFERS
    )

    # At most one offer per customer.
    customer_groups = options.groupby("customer_id").indices

    for indices in customer_groups.values():
        optimization.Add(
            sum(selected[int(i)] for i in indices)
            <= MAX_OFFERS_PER_CUSTOMER
        )

    # Maximize total estimated net value.
    optimization.Maximize(
        sum(
            int(options.loc[i, "objective_value"]) * selected[i]
            for i in range(len(options))
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0

    status = solver.Solve(optimization)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        raise RuntimeError(
            f"Optimization did not find a solution: "
            f"{solver.StatusName(status)}"
        )

    chosen_indices = [
        i for i in range(len(options))
        if solver.Value(selected[i]) == 1
    ]

    recommendations = options.iloc[
        chosen_indices
    ].copy()

    recommendations = recommendations.sort_values(
        "expected_net_value_inr",
        ascending=False,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    recommendations.to_csv(
        OUTPUT_FILE, index=False
    )

    total_spend = recommendations["offer_cost_inr"].sum()
    total_benefit = recommendations[
        "expected_benefit_inr"
    ].sum()
    total_net_value = recommendations[
        "expected_net_value_inr"
    ].sum()

    print("\nRetention optimization completed.")
    print("Solver status:", solver.StatusName(status))
    print("Customers scored:", len(df))
    print("Offers selected:", len(recommendations))
    print(f"Budget: INR {BUDGET_INR:,.0f}")
    print(f"Actual modeled spend: INR {total_spend:,.0f}")
    print(f"Budget remaining: INR {BUDGET_INR - total_spend:,.0f}")
    print(
        f"Assumed expected benefit: INR {total_benefit:,.0f}"
    )
    print(
        f"Assumed expected net value: INR {total_net_value:,.0f}"
    )
    print("Recommendations saved to:", OUTPUT_FILE)

    print("\nTop 10 recommendations:")
    print(
        recommendations[
            [
                "customer_id",
                "channel",
                "churn_probability",
                "offer_cost_inr",
                "expected_net_value_inr",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nWARNING: Results depend on illustrative assumptions. "
        "They are not validated business forecasts."
    )


if __name__ == "__main__":
    main()
