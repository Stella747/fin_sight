
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "generated" / "customer_360.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "churn_model.joblib"


def main():
    # 1. Load data.
    df = pd.read_csv(DATA_FILE)

    target = "churned"

    # Exclude IDs, the unrelated offer target, and the
    # churn label itself from the input features.
    excluded = {
        "customer_id",
        "churned",
        "offer_accepted_reference",
    }

    X = df.drop(columns=list(excluded))
    y = df[target].astype(int)

    print("Dataset shape:", df.shape)
    print("Features:", len(X.columns))
    print("Churn rate:", round(y.mean(), 4))

    # 2. Identify numerical and categorical columns.
    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Numerical features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    # 3. Split before fitting preprocessing.
    # Stratification preserves the approximate churn
    # percentage in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Build preprocessing pipelines.
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    # 5. Logistic Regression baseline.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Fit preprocessing and model on training data only.
    model.fit(X_train, y_train)

    # 6. Evaluate on the held-out test set.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- Test Results ---")
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
    print("Average precision:", round(
        average_precision_score(y_test, probabilities), 4
    ))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["No churn", "Churn"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # 7. Save the complete pipeline for later inference.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nModel saved to:", MODEL_FILE)


if __name__ == "__main__":
    main()
