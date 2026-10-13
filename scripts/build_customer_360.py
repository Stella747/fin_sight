
from pathlib import Path

import pandas as pd


# Locate the project directory automatically.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "generated"

EXCEL_FILE = DATA_DIR / "finsight_synthetic_data.xlsx"
OUTPUT_CSV = DATA_DIR / "customer_360.csv"


def main():
    # 1. Load the source tables.
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Workbook not found: {EXCEL_FILE}"
        )

    customers = pd.read_excel(
        EXCEL_FILE, sheet_name="customers"
    )
    usage = pd.read_excel(
        EXCEL_FILE, sheet_name="product_usage"
    )
    transactions = pd.read_excel(
        EXCEL_FILE, sheet_name="transactions"
    )

    print("Source tables loaded.")
    print("Customers:", customers.shape)
    print("Product usage:", usage.shape)
    print("Transactions:", transactions.shape)

    # 2. Validate customer identifiers.
    assert customers["customer_id"].is_unique, (
        "Duplicate customer IDs found."
    )
    assert usage["customer_id"].is_unique, (
        "Duplicate product usage customer IDs found."
    )
    assert transactions["transaction_id"].is_unique, (
        "Duplicate transaction IDs found."
    )

    customer_ids = set(customers["customer_id"])

    assert set(usage["customer_id"]).issubset(
        customer_ids
    ), "Product usage contains unknown customers."

    assert set(transactions["customer_id"]).issubset(
        customer_ids
    ), "Transactions contain unknown customers."

    # 3. Create transaction aggregates.
    #
    # Grouping converts transaction-level data into
    # customer-level behavioral features.

    transaction_features = (
        transactions.groupby("customer_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            total_transaction_amount_inr=(
                "amount_inr", "sum"
            ),
            average_transaction_amount_inr=(
                "amount_inr", "mean"
            ),
            last_transaction_date=(
                "transaction_date", "max"
            ),
        )
        .reset_index()
    )

    # Derive recency relative to a fixed reference date.
    # The source generator uses dates up to 2026-10-01.
    reference_date = pd.Timestamp("2026-10-02")

    transaction_features["last_transaction_date"] = (
        pd.to_datetime(
            transaction_features["last_transaction_date"]
        )
    )

    transaction_features["transaction_recency_days"] = (
        reference_date
        - transaction_features["last_transaction_date"]
    ).dt.days

    # The raw date is not required for this first feature table.
    transaction_features = transaction_features.drop(
        columns=["last_transaction_date"]
    )

    # 4. Join the tables using customer_id.
    customer_360 = customers.merge(
        usage,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    customer_360 = customer_360.merge(
        transaction_features,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # 5. Handle customers without transactions.
    # Count and monetary aggregates are zero when no
    # matching transactions exist. Recency stays missing
    # because "no transaction" is not the same as a
    # recent transaction.

    aggregate_columns = [
        "transaction_count",
        "total_transaction_amount_inr",
        "average_transaction_amount_inr",
    ]

    customer_360[aggregate_columns] = (
        customer_360[aggregate_columns].fillna(0)
    )

    # 6. Validate the final dataset.
    assert len(customer_360) == len(customers), (
        "Customer row count changed after merging."
    )

    assert customer_360["customer_id"].is_unique, (
        "Customer 360 has duplicate customer IDs."
    )

    required_columns = [
        "customer_id",
        "monthly_transactions",
        "churned",
        "transaction_count",
        "transaction_recency_days",
    ]

    assert customer_360[required_columns].notna().all().all(), (
        "Missing values found in required columns."
    )

    # 7. Save the feature table.
    customer_360.to_csv(OUTPUT_CSV, index=False)

    print("\nCustomer 360 created successfully!")
    print("Output:", OUTPUT_CSV)
    print("Shape:", customer_360.shape)
    print(
        "Unique customers:",
        customer_360["customer_id"].nunique(),
    )
    print(
        "Missing values in all columns:",
        int(customer_360.isna().sum().sum()),
    )

    print("\nPreview:")
    print(customer_360.head().to_string(index=False))

    print("\nChurn label distribution:")
    print(
        customer_360["churned"]
        .value_counts(normalize=True)
        .rename("proportion")
    )


if __name__ == "__main__":
    main()
