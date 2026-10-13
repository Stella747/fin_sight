
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / "data" / "generated" / "customer_360.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Customer 360 file not found: {CSV_FILE}"
    )

# Create an in-memory analytical database.
con = duckdb.connect(database=":memory:")

# Register the CSV as a queryable view.
# DuckDB reads the file when executing SQL queries.
csv_path = CSV_FILE.as_posix().replace("'", "''")

con.execute(f"""
    CREATE VIEW customer_360 AS
    SELECT *
    FROM read_csv_auto('{csv_path}')
""")

print("\n1. Dataset overview")
print(
    con.execute("""
        SELECT
            COUNT(*) AS customer_count,
            COUNT(DISTINCT customer_id) AS unique_customers,
            ROUND(AVG(churned), 4) AS churn_rate,
            ROUND(AVG(monthly_transactions), 2)
                AS avg_monthly_transactions
        FROM customer_360
    """).df().to_string(index=False)
)

print("\n2. Churn by customer segment")
print(
    con.execute("""
        SELECT
            customer_segment,
            COUNT(*) AS customers,
            ROUND(AVG(churned), 4) AS churn_rate
        FROM customer_360
        GROUP BY customer_segment
        ORDER BY churn_rate DESC
    """).df().to_string(index=False)
)

print("\n3. Transaction activity by churn outcome")
print(
    con.execute("""
        SELECT
            churned,
            COUNT(*) AS customers,
            ROUND(AVG(monthly_transactions), 2)
                AS avg_monthly_transactions,
            ROUND(AVG(app_logins_last_30d), 2)
                AS avg_app_logins,
            ROUND(AVG(support_tickets_last_90d), 2)
                AS avg_support_tickets
        FROM customer_360
        GROUP BY churned
        ORDER BY churned
    """).df().to_string(index=False)
)

print("\n4. Customers with highest transaction amounts")
print(
    con.execute("""
        SELECT
            customer_id,
            customer_segment,
            transaction_count,
            ROUND(total_transaction_amount_inr, 2)
                AS total_transaction_amount_inr
        FROM customer_360
        ORDER BY total_transaction_amount_inr DESC
        LIMIT 10
    """).df().to_string(index=False)
)

con.close()
print("\nAnalytics completed.")
