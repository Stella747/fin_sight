
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Reproducibility and file paths
# --------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "finsight_synthetic_data.xlsx"

N_CUSTOMERS = 5_000
N_TRANSACTIONS = 50_000
N_CAMPAIGNS = 10_000


def sigmoid(x):
    """Convert scores into probabilities between 0 and 1."""
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------
# 2. Customer master dataset
# --------------------------------------------------

customer_ids = np.array(
    [f"C{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
)

age = rng.integers(21, 71, N_CUSTOMERS)
tenure_years = np.round(
    rng.uniform(0.1, 15, N_CUSTOMERS), 1
)

income = np.clip(
    rng.lognormal(mean=np.log(65000), sigma=0.45,
                  size=N_CUSTOMERS),
    18000, 300000
).round(0)

segment = rng.choice(
    ["Mass", "Affluent", "Premium"],
    size=N_CUSTOMERS,
    p=[0.60, 0.30, 0.10]
)

region = rng.choice(
    ["North", "South", "East", "West", "Central"],
    size=N_CUSTOMERS
)

digital_preference = rng.choice(
    ["Low", "Medium", "High"],
    size=N_CUSTOMERS,
    p=[0.20, 0.45, 0.35]
)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "age": age,
    "tenure_years": tenure_years,
    "annual_income_inr": (income * 12).astype(int),
    "customer_segment": segment,
    "region": region,
    "digital_preference": digital_preference,
})


# --------------------------------------------------
# 3. Product usage dataset
# --------------------------------------------------

monthly_transactions = np.clip(
    rng.poisson(lam=35, size=N_CUSTOMERS), 0, 150
)

avg_balance = np.clip(
    income * rng.uniform(1.0, 8.0, N_CUSTOMERS),
    1000, 5_000_000
).round(2)

credit_utilization = rng.beta(
    2, 5, N_CUSTOMERS
).round(3)

app_logins = np.clip(
    rng.poisson(
        lam=np.where(
            digital_preference == "High", 18,
            np.where(digital_preference == "Medium", 10, 3)
        ),
        size=N_CUSTOMERS
    ),
    0, 90
)

support_tickets = np.clip(
    rng.poisson(1.2, N_CUSTOMERS), 0, 12
)

# Higher values mean more recent inactivity.
days_since_last_transaction = np.clip(
    rng.exponential(8, N_CUSTOMERS), 0, 90
).round(1)

salary_growth = rng.normal(
    0.05, 0.12, N_CUSTOMERS
).clip(-0.35, 0.60).round(3)

product_usage = pd.DataFrame({
    "customer_id": customer_ids,
    "monthly_transactions": monthly_transactions,
    "average_balance_inr": avg_balance,
    "credit_utilization": credit_utilization,
    "app_logins_last_30d": app_logins,
    "support_tickets_last_90d": support_tickets,
    "days_since_last_transaction": days_since_last_transaction,
    "salary_growth_rate": salary_growth,
})


# --------------------------------------------------
# 4. Synthetic churn outcomes
# --------------------------------------------------

# These scores simulate plausible relationships.
# They are not calibrated to real banking data.

churn_score = (
    -2.0
    + 0.035 * days_since_last_transaction
    - 0.025 * monthly_transactions
    + 0.18 * support_tickets
    - 0.015 * app_logins
    + 0.8 * (digital_preference == "Low")
)

churn_probability = sigmoid(churn_score)
churned = rng.binomial(1, churn_probability)

customers["churned"] = churned


# --------------------------------------------------
# 5. Synthetic offer acceptance outcomes
# --------------------------------------------------

offer_score = (
    -1.0
    + 0.012 * monthly_transactions
    + 0.7 * (digital_preference == "High")
    + 0.8 * (segment == "Affluent")
    + 1.2 * salary_growth
    - 1.5 * credit_utilization
)

offer_probability = sigmoid(offer_score)
offer_accepted = rng.binomial(1, offer_probability)

# Keep this label on the customer table as a simple
# initial modeling target. Campaign outcomes are
# generated separately below.
customers["offer_accepted_reference"] = offer_accepted


# --------------------------------------------------
# 6. Transaction-level dataset
# --------------------------------------------------

transaction_customer_idx = rng.integers(
    0, N_CUSTOMERS, N_TRANSACTIONS
)

transaction_dates = (
    pd.Timestamp("2026-10-01")
    - pd.to_timedelta(
        rng.integers(0, 180, N_TRANSACTIONS), unit="D"
    )
)

transactions = pd.DataFrame({
    "transaction_id": [
        f"T{i:08d}" for i in range(1, N_TRANSACTIONS + 1)
    ],
    "customer_id": customer_ids[transaction_customer_idx],
    "transaction_date": transaction_dates,
    "transaction_type": rng.choice(
        ["Purchase", "Deposit", "Transfer", "Bill Payment"],
        size=N_TRANSACTIONS,
        p=[0.45, 0.20, 0.20, 0.15]
    ),
    "amount_inr": np.round(
        rng.lognormal(np.log(1800), 1.0, N_TRANSACTIONS),
        2
    ),
    "channel": rng.choice(
        ["Mobile App", "Web", "ATM", "Branch"],
        size=N_TRANSACTIONS,
        p=[0.55, 0.20, 0.15, 0.10]
    ),
})

transactions = transactions.sort_values(
    "transaction_date"
).reset_index(drop=True)


# --------------------------------------------------
# 7. Campaign history dataset
# --------------------------------------------------

campaign_customer_idx = rng.integers(
    0, N_CUSTOMERS, N_CAMPAIGNS
)

campaign_type = rng.choice(
    ["Investment", "Cashback", "Loan", "Credit Card"],
    size=N_CAMPAIGNS
)

campaign_channel = rng.choice(
    ["Mobile App", "Email", "SMS", "Phone"],
    size=N_CAMPAIGNS,
    p=[0.40, 0.30, 0.20, 0.10]
)

campaign_cost = rng.choice(
    [5, 10, 20, 50, 100],
    size=N_CAMPAIGNS
).astype(float)

campaign_customers = customers.iloc[
    campaign_customer_idx
].reset_index(drop=True)

campaign_usage = product_usage.iloc[
    campaign_customer_idx
].reset_index(drop=True)

campaign_segment = campaign_customers["customer_segment"].to_numpy()
campaign_digital = campaign_customers["digital_preference"].to_numpy()

campaign_score = (
    -1.2
    + 0.015 * campaign_usage["monthly_transactions"].to_numpy()
    + 0.7 * (campaign_digital == "High")
    + 0.6 * (campaign_segment == "Affluent")
    + 0.3 * (campaign_type == "Investment")
    + 0.2 * (campaign_channel == "Mobile App")
)

campaign_probability = sigmoid(campaign_score)
campaign_response = rng.binomial(1, campaign_probability)

campaign_history = pd.DataFrame({
    "campaign_id": [
        f"CAM{i:07d}" for i in range(1, N_CAMPAIGNS + 1)
    ],
    "customer_id": customer_ids[campaign_customer_idx],
    "campaign_date": (
        pd.Timestamp("2026-10-01")
        - pd.to_timedelta(
            rng.integers(0, 180, N_CAMPAIGNS), unit="D"
        )
    ),
    "campaign_type": campaign_type,
    "channel": campaign_channel,
    "campaign_cost_inr": campaign_cost,
    "responded": campaign_response,
})


# --------------------------------------------------
# 8. Data dictionary
# --------------------------------------------------

dictionary_rows = [
    ("customers", "customer_id", "Unique customer identifier"),
    ("customers", "age", "Customer age in years"),
    ("customers", "tenure_years", "Account tenure in years"),
    ("customers", "annual_income_inr", "Synthetic annual income in INR"),
    ("customers", "customer_segment", "Synthetic customer segment"),
    ("customers", "region", "Synthetic geographic region"),
    ("customers", "digital_preference", "Synthetic digital engagement category"),
    ("customers", "churned", "Synthetic binary churn outcome: 1 or 0"),
    ("customers", "offer_accepted_reference", "Synthetic binary offer outcome"),
    ("product_usage", "monthly_transactions", "Transactions in a typical month"),
    ("product_usage", "average_balance_inr", "Synthetic average account balance"),
    ("product_usage", "credit_utilization", "Credit utilization ratio from 0 to 1"),
    ("product_usage", "app_logins_last_30d", "App logins in the last 30 days"),
    ("product_usage", "support_tickets_last_90d", "Support tickets in the last 90 days"),
    ("product_usage", "days_since_last_transaction", "Days since last transaction"),
    ("product_usage", "salary_growth_rate", "Synthetic salary growth rate"),
    ("transactions", "transaction_id", "Unique transaction identifier"),
    ("transactions", "customer_id", "Customer associated with transaction"),
    ("transactions", "transaction_date", "Synthetic transaction date"),
    ("transactions", "transaction_type", "Synthetic transaction category"),
    ("transactions", "amount_inr", "Synthetic transaction amount in INR"),
    ("transactions", "channel", "Synthetic transaction channel"),
    ("campaign_history", "campaign_id", "Unique campaign record identifier"),
    ("campaign_history", "customer_id", "Customer receiving the campaign"),
    ("campaign_history", "campaign_date", "Synthetic campaign date"),
    ("campaign_history", "campaign_type", "Product offer category"),
    ("campaign_history", "channel", "Campaign communication channel"),
    ("campaign_history", "campaign_cost_inr", "Synthetic campaign cost in INR"),
    ("campaign_history", "responded", "Synthetic campaign response: 1 or 0"),
]

data_dictionary = pd.DataFrame(
    dictionary_rows,
    columns=["table_name", "column_name", "description"]
)


# --------------------------------------------------
# 9. Basic data-quality checks
# --------------------------------------------------

assert customers["customer_id"].is_unique
assert transactions["transaction_id"].is_unique
assert campaign_history["campaign_id"].is_unique

assert set(transactions["customer_id"]).issubset(
    set(customers["customer_id"])
)
assert set(campaign_history["customer_id"]).issubset(
    set(customers["customer_id"])
)

assert customers["churned"].isin([0, 1]).all()
assert campaign_history["responded"].isin([0, 1]).all()


# --------------------------------------------------
# 10. Export Excel workbook
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(
    OUTPUT_FILE, engine="openpyxl"
) as writer:
    customers.to_excel(
        writer, sheet_name="customers", index=False
    )
    transactions.to_excel(
        writer, sheet_name="transactions", index=False
    )
    product_usage.to_excel(
        writer, sheet_name="product_usage", index=False
    )
    campaign_history.to_excel(
        writer, sheet_name="campaign_history", index=False
    )
    data_dictionary.to_excel(
        writer, sheet_name="data_dictionary", index=False
    )

print("\nFinSight synthetic dataset generated successfully!")
print(f"Output: {OUTPUT_FILE}")
print(f"Customers: {len(customers):,}")
print(f"Transactions: {len(transactions):,}")
print(f"Product usage records: {len(product_usage):,}")
print(f"Campaign records: {len(campaign_history):,}")
print(f"Churn rate: {customers['churned'].mean():.1%}")
print(f"Reference offer acceptance: {customers['offer_accepted_reference'].mean():.1%}")
print(f"Campaign response rate: {campaign_history['responded'].mean():.1%}")
