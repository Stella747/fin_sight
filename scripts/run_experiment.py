
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "generated"

INPUT_FILE = DATA_DIR / "customer_360.csv"
EXPERIMENT_FILE = DATA_DIR / "retention_experiment.csv"
RESULT_FILE = DATA_DIR / "experiment_results.csv"

SEED = 42
TREATMENT_EFFECT = 0.03  # Simulated +3 percentage points


def wilson_interval(successes, total, confidence=0.95):
    """Wilson confidence interval for a single proportion."""
    if total == 0:
        return (np.nan, np.nan)

    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return center - margin, center + margin


def main():
    rng = np.random.default_rng(SEED)

    customers = pd.read_csv(INPUT_FILE)

    # One experimental record per customer.
    experiment = customers[
        ["customer_id", "customer_segment"]
    ].copy()

    n = len(experiment)

    # Random assignment: 0 = control, 1 = treatment.
    assignment = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(assignment)

    experiment["treatment"] = assignment

    # Simulated outcome generation.
    # This deliberately creates a known average treatment
    # effect of approximately +3 percentage points.
    baseline_probability = 0.15

    probabilities = (
        baseline_probability
        + TREATMENT_EFFECT * experiment["treatment"].to_numpy()
    )

    experiment["converted"] = rng.binomial(
        1, probabilities
    )

    # Save the experiment-level data.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    experiment.to_csv(EXPERIMENT_FILE, index=False)

    # Calculate group statistics.
    summary = (
        experiment.groupby("treatment")
        .agg(
            customers=("customer_id", "count"),
            conversions=("converted", "sum"),
            conversion_rate=("converted", "mean"),
        )
        .reset_index()
    )

    control = summary.loc[
        summary["treatment"] == 0
    ].iloc[0]

    treatment = summary.loc[
        summary["treatment"] == 1
    ].iloc[0]

    n_control = int(control["customers"])
    n_treatment = int(treatment["customers"])

    x_control = int(control["conversions"])
    x_treatment = int(treatment["conversions"])

    p_control = float(control["conversion_rate"])
    p_treatment = float(treatment["conversion_rate"])

    # Absolute lift, expressed as a proportion.
    absolute_lift = p_treatment - p_control

    # Relative lift; undefined when the control rate is zero.
    relative_lift = (
        absolute_lift / p_control
        if p_control > 0 else np.nan
    )

    # Unpooled Wald confidence interval for the difference
    # between two independent proportions.
    standard_error = np.sqrt(
        p_control * (1 - p_control) / n_control
        + p_treatment * (1 - p_treatment) / n_treatment
    )

    ci_lower = absolute_lift - 1.96 * standard_error
    ci_upper = absolute_lift + 1.96 * standard_error

    # Two-proportion z-test using the pooled null estimate.
    pooled_rate = (
        x_control + x_treatment
    ) / (n_control + n_treatment)

    pooled_se = np.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (1 / n_control + 1 / n_treatment)
    )

    z_score = (
        absolute_lift / pooled_se
        if pooled_se > 0 else 0.0
    )

    p_value = 2 * norm.sf(abs(z_score))

    control_ci = wilson_interval(
        x_control, n_control
    )
    treatment_ci = wilson_interval(
        x_treatment, n_treatment
    )

    results = pd.DataFrame([{
        "control_customers": n_control,
        "control_conversions": x_control,
        "control_rate": p_control,
        "control_ci_lower": control_ci[0],
        "control_ci_upper": control_ci[1],
        "treatment_customers": n_treatment,
        "treatment_conversions": x_treatment,
        "treatment_rate": p_treatment,
        "treatment_ci_lower": treatment_ci[0],
        "treatment_ci_upper": treatment_ci[1],
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "lift_ci_lower": ci_lower,
        "lift_ci_upper": ci_upper,
        "z_score": z_score,
        "p_value": p_value,
    }])

    results.to_csv(RESULT_FILE, index=False)

    print("\nRetention experiment completed.")
    print("\nGroup summary:")
    print(summary.to_string(index=False))

    print(f"\nControl conversion rate: {p_control:.2%}")
    print(f"Treatment conversion rate: {p_treatment:.2%}")
    print(f"Absolute lift: {absolute_lift:+.2%}")
    print(
        f"Relative lift: {relative_lift:+.2%}"
        if np.isfinite(relative_lift)
        else "Relative lift: undefined"
    )
    print(
        f"95% CI for absolute lift: "
        f"[{ci_lower:+.2%}, {ci_upper:+.2%}]"
    )
    print(f"Z-score: {z_score:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Result: statistically significant at 5%.")
    else:
        print("Result: not statistically significant at 5%.")

    print("\nFiles saved:")
    print(EXPERIMENT_FILE)
    print(RESULT_FILE)

    print(
        "\nIMPORTANT: This is a simulated experiment. "
        "The treatment effect is an assumption built into "
        "the synthetic outcome generator, not evidence "
        "that a real retention offer works."
    )


if __name__ == "__main__":
    main()
