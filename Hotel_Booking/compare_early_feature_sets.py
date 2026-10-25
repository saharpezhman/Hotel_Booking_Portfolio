from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_canceled"


# =====================================================
# 2. Feature sets
# =====================================================

# The existing model: 19 features
CURRENT_NUMERIC = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "adr",
]

CURRENT_CATEGORICAL = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
]

# A conservative candidate for early prediction
SAFE_NUMERIC = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "babies",
]

SAFE_CATEGORICAL = [
    "hotel",
    "meal",
    "reserved_room_type",
]


# =====================================================
# 3. Estimate booking date
# =====================================================

def estimate_booking_date(df):
    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    months = df["arrival_date_month"].map(month_map)

    if months.isna().any():
        bad_months = df.loc[
            months.isna(), "arrival_date_month"
        ].drop_duplicates().tolist()

        raise ValueError(f"Unknown arrival month values: {bad_months}")

    arrival_date = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": months,
        "day": df["arrival_date_day_of_month"],
    })

    # This is an estimate, not a recorded booking timestamp.
    booking_date = (
        arrival_date
        - pd.to_timedelta(df["lead_time"], unit="D")
    )

    return booking_date


# =====================================================
# 4. Build a model pipeline
# =====================================================

def build_model(numeric_features, categorical_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown",
            ),
        ),
        (
            "to_string",
            FunctionTransformer(
                lambda X: X.astype(str)
            ),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ])

    return model


# =====================================================
# 5. Evaluate a model
# =====================================================

def evaluate_model(
    name,
    df,
    train_mask,
    test_mask,
    numeric_features,
    categorical_features,
):

    features = numeric_features + categorical_features

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]

    y_train = df.loc[train_mask, TARGET].astype(int)
    y_test = df.loc[test_mask, TARGET].astype(int)

    model = build_model(
        numeric_features,
        categorical_features,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "model": name,
        "feature_count": len(features),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return results


# =====================================================
# 6. Main
# =====================================================

def main():

    print("=" * 70)
    print("COMPARISON: CURRENT VS CONSERVATIVE EARLY MODEL")
    print("=" * 70)

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)

    required_columns = set(
        CURRENT_NUMERIC
        + CURRENT_CATEGORICAL
        + ["is_canceled"]
        + [
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_day_of_month",
        ]
    )

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Use the same chronological splitting approach as before.
    df["_estimated_booking_date"] = estimate_booking_date(df)

    unique_dates = sorted(
        df["_estimated_booking_date"].dropna().unique()
    )

    split_index = int(len(unique_dates) * 0.80)
    cutoff_date = pd.Timestamp(unique_dates[split_index])

    train_mask = df["_estimated_booking_date"] < cutoff_date
    test_mask = df["_estimated_booking_date"] >= cutoff_date

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Temporal split produced an empty dataset.")

    print(f"\nCutoff date: {cutoff_date.date()}")
    print(f"Training rows: {int(train_mask.sum()):,}")
    print(f"Testing rows: {int(test_mask.sum()):,}")
    print(
        "Train cancellation rate: "
        f"{df.loc[train_mask, TARGET].mean():.4f}"
    )
    print(
        "Test cancellation rate: "
        f"{df.loc[test_mask, TARGET].mean():.4f}"
    )

    print("\nTraining current 19-feature model...")

    current_results = evaluate_model(
        "Current model - 19 features",
        df,
        train_mask,
        test_mask,
        CURRENT_NUMERIC,
        CURRENT_CATEGORICAL,
    )

    print("Training conservative 8-feature model...")

    conservative_results = evaluate_model(
        "Conservative model - 8 features",
        df,
        train_mask,
        test_mask,
        SAFE_NUMERIC,
        SAFE_CATEGORICAL,
    )

    results = pd.DataFrame([
        current_results,
        conservative_results,
    ])

    print("\n" + "=" * 70)
    print("TEMPORAL TEST RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "model",
                "feature_count",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,"roc_auc": "{:.4f}".format,
            },
        )
    )

    output_file = (
        RESULTS_DIR / "early_feature_set_comparison.csv"
    )

    results.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nComparison report saved to:\n{output_file}")

    print("\nIMPORTANT:")
    print(
        "Booking dates are estimated from arrival date and lead time. "
        "The conservative model is a candidate for comparison, not "
        "proof that every included feature was available at booking time."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()