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

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "is_canceled"

NUMERIC = [
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

CATEGORICAL = [
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

GROUPS = {
    "All 19 features": (NUMERIC, CATEGORICAL),

    "Without customer history": (
        [
            f for f in NUMERIC
            if f not in [
                "is_repeated_guest",
                "previous_cancellations",
                "previous_bookings_not_canceled",
            ]
        ],
        CATEGORICAL,
    ),

    "Without ADR and agent": (
        [f for f in NUMERIC if f != "adr"],
        [f for f in CATEGORICAL if f != "agent"],
    ),

    "Without all unverified features": (
        [
            "lead_time",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "babies",
        ],
        [
            "hotel",
            "meal",
            "reserved_room_type",
        ],
    ),
}


def estimate_booking_date(df):
    month_map = {
        "January": 1, "February": 2, "March": 3,
        "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9,
        "October": 10, "November": 11, "December": 12,
    }

    months = df["arrival_date_month"].map(month_map)

    if months.isna().any():
        raise ValueError("Unknown or missing arrival month values.")

    arrival_date = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": months,
        "day": df["arrival_date_day_of_month"],
    })

    return arrival_date - pd.to_timedelta(
        df["lead_time"], unit="D"
    )


def build_model(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="Unknown"),
        ),
        (
            "to_string",
            FunctionTransformer(lambda X: X.astype(str)),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
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


def main():
    print("=" * 70)
    print("FEATURE GROUP ABLATION - TEMPORAL EVALUATION")
    print("=" * 70)

    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    required = set(
        NUMERIC + CATEGORICAL + [
            TARGET,
            "arrival_date_year",
            "arrival_date_month",
            "arrival_date_day_of_month",
        ]
    )

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["_estimated_booking_date"] = estimate_booking_date(df)

    unique_dates = sorted(
        df["_estimated_booking_date"].dropna().unique()
    )

    cutoff_index = int(len(unique_dates) * 0.80)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_index])

    train_mask = df["_estimated_booking_date"] < cutoff_date
    test_mask = df["_estimated_booking_date"] >= cutoff_date

    y_train = df.loc[train_mask, TARGET].astype(int)
    y_test = df.loc[test_mask, TARGET].astype(int)

    print(f"\nCutoff date: {cutoff_date.date()}")
    print(f"Training rows: {len(y_train):,}")
    print(f"Testing rows: {len(y_test):,}")

    rows = []

    for name, (numeric_features, categorical_features) in GROUPS.items():
        features = numeric_features + categorical_features

        print(f"\nTraining: {name} ({len(features)} features)")

        model = build_model(numeric_features, categorical_features)

        model.fit(
            df.loc[train_mask, features],
            y_train,
        )

        predictions = model.predict(
            df.loc[test_mask, features]
        )
        probabilities = model.predict_proba(
            df.loc[test_mask, features]
        )[:, 1]

        rows.append({
            "model": name,
            "feature_count": len(features),
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
        })

    results = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(results.to_string(
        index=False,
        formatters={
            metric: "{:.4f}".format
            for metric in [
                "accuracy", "precision", "recall", "f1", "roc_auc"
            ]
        },
    ))

    output_file = RESULTS_DIR / "feature_group_ablation.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nSaved report:\n{output_file}")
    print(
        "\nNote: booking dates are estimated. This experiment compares "
        "feature groups but does not prove their real-time availability."
    )


if __name__ == "__main__":
    main()