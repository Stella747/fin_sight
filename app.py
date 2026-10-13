
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "generated"
REPORTS = ROOT / "reports" / "shap"
MODEL_FILE = ROOT / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="FinSight | Financial Decision Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("FinSight")
st.subheader("AI Financial Decision Intelligence Platform")

st.warning(
    "Educational demo using synthetic data. Predictions, "
    "retention values and experiment outcomes are simulated "
    "and must not be used for real financial decisions."
)


@st.cache_data
def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


customers = load_csv("customer_360.csv")
recommendations = load_csv("retention_recommendations.csv")
experiment_results = load_csv("experiment_results.csv")
coefficients = load_csv("churn_model_coefficients.csv")

if customers is None:
    st.error(
        "Customer 360 is missing. Run "
        "`python scripts/build_customer_360.py` first."
    )
    st.stop()

required = {
    "customer_id",
    "customer_segment",
    "churned",
}

if not required.issubset(customers.columns):
    st.error("Customer 360 is missing required columns.")
    st.stop()


# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Customer Intelligence",
        "Retention Optimization",
        "Experimentation",
        "Model Explainability",
    ],
)

st.sidebar.divider()
st.sidebar.caption("FinSight | Synthetic banking data")


# --------------------------------------------------
# 1. Executive Overview
# --------------------------------------------------

if page == "Executive Overview":
    st.header("Executive Overview")

    total_customers = len(customers)
    churn_rate = customers["churned"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{total_customers:,}")
    col2.metric("Synthetic churn rate", f"{churn_rate:.2%}")

    if recommendations is not None:
        col3.metric(
            "Offers selected",
            f"{len(recommendations):,}",
        )
        spend = recommendations["offer_cost_inr"].sum()
        col4.metric("Selected offer spend", f"₹{spend:,.0f}")
    else:
        col3.metric("Offers selected", "Unavailable")
        col4.metric("Selected offer spend", "Unavailable")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customers by segment")
        segment_counts = (
            customers["customer_segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )
        fig = px.bar(
            segment_counts,
            x="Segment",
            y="Customers",
            title="Customer distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn rate by segment")
        segment_churn = (
            customers.groupby("customer_segment", as_index=False)
            .agg(
                customers=("customer_id", "count"),
                churn_rate=("churned", "mean"),
            )
        )
        segment_churn["churn_rate_pct"] = (
            segment_churn["churn_rate"] * 100
        )

        fig = px.bar(
            segment_churn,
            x="customer_segment",
            y="churn_rate_pct",
            labels={
                "customer_segment": "Segment",
                "churn_rate_pct": "Churn rate (%)",
            },
            title="Synthetic churn rate by segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Churn rates describe generated labels, not real-world "
        "banking performance."
    )


# --------------------------------------------------
# 2. Customer Intelligence
# --------------------------------------------------

elif page == "Customer Intelligence":
    st.header("Customer Intelligence")

    model = load_model()

    if model is None:
        st.error("Trained model not found in models/.")
        st.stop()

    customer_id = st.selectbox(
        "Select a customer",
        customers["customer_id"].tolist(),
    )

    customer = customers.loc[
        customers["customer_id"] == customer_id
    ].iloc[0]

    excluded = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }
    feature_columns = [
        column for column in customers.columns
        if column not in excluded
    ]

    features = customers.loc[
        customers["customer_id"] == customer_id,
        feature_columns,
    ]

    probability = model.predict_proba(features)[0, 1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer ID", customer_id)
    col2.metric("Segment", customer["customer_segment"])
    col3.metric("Model churn probability", f"{probability:.1%}")

    st.progress(float(probability))

    if "monthly_transactions" in customers.columns:
        st.write(
            "Monthly transactions:",
            int(customer["monthly_transactions"]),
        )

    if "average_balance_inr" in customers.columns:
        st.write(
            "Average balance:",
            f"₹{customer['average_balance_inr']:,.2f}",
        )

    if "app_logins_last_30d" in customers.columns:
        st.write(
            "App logins in last 30 days:",
            int(customer["app_logins_last_30d"]),
        )

    if "support_tickets_last_90d" in customers.columns:
        st.write(
            "Support tickets in last 90 days:",
            int(customer["support_tickets_last_90d"]),
        )

    st.info(
        "This is a synthetic model score, not a calibrated risk "
        "estimate for a real customer."
    )


# --------------------------------------------------
# 3. Retention Optimization
# --------------------------------------------------

elif page == "Retention Optimization":
    st.header("Retention Optimization")

    if recommendations is None:
        st.error(
            "Recommendations are missing. Run "
            "`python scripts/optimize_retention.py`."
        )
        st.stop()

    budget = 100_000
    spend = recommendations["offer_cost_inr"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Offers selected", f"{len(recommendations):,}")
    col2.metric("Spend", f"₹{spend:,.0f}")
    col3.metric("Budget remaining", f"₹{budget - spend:,.0f}")

    if "channel" in recommendations.columns:
        channel_summary = (
            recommendations.groupby("channel")
            .size()
            .reset_index(name="Offers")
        )

        fig = px.bar(
            channel_summary,
            x="channel",
            y="Offers",
            title="Selected offers by channel",
        )
        st.plotly_chart(fig, use_container_width=True)

    display_columns = [
        column for column in [
            "customer_id",
            "channel",
            "churn_probability",
            "offer_cost_inr",
            "expected_net_value_inr",
        ]
        if column in recommendations.columns
    ]

    st.subheader("Recommended offers")
    st.dataframe(
        recommendations[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download recommendations CSV",
        data=recommendations.to_csv(index=False).encode("utf-8"),
        file_name="retention_recommendations.csv",
        mime="text/csv",
    )

    st.warning(
        "Expected benefit and net value depend on illustrative "
        "assumptions, not measured incremental retention."
    )


# --------------------------------------------------
# 4. Experimentation
# --------------------------------------------------

elif page == "Experimentation":
    st.header("Experimentation Results")

    if experiment_results is None:
        st.error(
            "Experiment results are missing. Run "
            "`python scripts/run_experiment.py`."
        )
        st.stop()

    result = experiment_results.iloc[0]

    control_rate = result["control_rate"]
    treatment_rate = result["treatment_rate"]
    lift = result["absolute_lift"]
    p_value = result["p_value"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Control conversion", f"{control_rate:.2%}")
    col2.metric("Treatment conversion", f"{treatment_rate:.2%}")
    col3.metric(
        "Absolute lift",
        f"{lift * 100:+.2f} pp",
    )

    chart_data = pd.DataFrame({
        "Group": ["Control", "Treatment"],
        "Conversion rate (%)": [
            control_rate * 100,
            treatment_rate * 100,
        ],
    })

    fig = px.bar(
        chart_data,
        x="Group",
        y="Conversion rate (%)",
        title="Control vs treatment conversion",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"Relative lift: {result['relative_lift']:.2%}")
    st.write(f"95% lift confidence interval: "
             f"{result['lift_ci_lower'] * 100:+.2f} to "
             f"{result['lift_ci_upper'] * 100:+.2f} pp")
    st.write(f"P-value: {p_value:.4f}")

    st.info(
        "The experiment is simulated with a preselected treatment "
        "effect. Statistical significance here does not validate "
        "the effectiveness of a real banking campaign."
    )


# --------------------------------------------------
# 5. Model Explainability
# --------------------------------------------------

elif page == "Model Explainability":
    st.header("Model Explainability")

    if coefficients is not None:
        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(20, len(coefficients)),
            value=min(10, len(coefficients)),
        )

        top = coefficients.head(top_n).copy()

        top["feature_label"] = top["feature"].str.replace(
            r"^(numeric|categorical)__",
            "",
            regex=True,
        )

        fig = px.bar(
            top.sort_values("coefficient"),
            x="coefficient",
            y="feature_label",
            orientation="h",
            title="Logistic Regression coefficients",
            labels={
                "coefficient": "Coefficient",
                "feature_label": "Feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Positive coefficients increase the model's log-odds "
            "score for churn; negative coefficients decrease it."
        )
    else:
        st.info("Coefficient explanation CSV not found.")

    for filename in [
        "global_feature_importance.png",
        "global_shap_summary.png",
        "individual_customer_explanation.png",
    ]:
        path = REPORTS / filename
        if path.exists():
            st.subheader(filename.replace(".png", "").replace("_", " ").title())
            st.image(str(path), use_container_width=True)

    st.warning(
        "Explanations describe model behavior on synthetic data; "
        "they do not establish causality."
    )
