from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# --------------------------------------------------
# 1. Paths and data
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 2. Estimate booking dates
# --------------------------------------------------

month_map = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12,
}

required_date_columns = [
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "lead_time",
    "is_canceled",
]

missing = [c for c in required_date_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

arrival_month = df["arrival_date_month"].map(month_map)

arrival_date = pd.to_datetime(
    {
        "year": df["arrival_date_year"],
        "month": arrival_month,
        "day": df["arrival_date_day_of_month"],
    },
    errors="coerce",
)

df["estimated_booking_date"] = (
    arrival_date - pd.to_timedelta(df["lead_time"], unit="D")
)

df = df.dropna(
    subset=["estimated_booking_date", "is_canceled"]
).copy()

df["is_canceled"] = df["is_canceled"].astype(int)
df = df.sort_values("estimated_booking_date").reset_index(drop=True)


# --------------------------------------------------
# 3. Features
# --------------------------------------------------

numeric_features = [
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

categorical_features = [
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

features = numeric_features + categorical_features

missing_features = [c for c in features if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X = df[features].copy()
y = df["is_canceled"].copy()


# --------------------------------------------------
# 4. Temporal split: keep each date in one set
# --------------------------------------------------

unique_dates = sorted(df["estimated_booking_date"].unique())

cutoff_index = int(len(unique_dates) * 0.80)
cutoff_index = max(1, min(cutoff_index, len(unique_dates) - 1))
cutoff_date = pd.Timestamp(unique_dates[cutoff_index])

train_mask = df["estimated_booking_date"] < cutoff_date
test_mask = df["estimated_booking_date"] >= cutoff_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("=" * 65)
print("BUSINESS EVALUATION - HOTEL CANCELLATION")
print("=" * 65)
print("Cutoff date:", cutoff_date)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 5. Preprocessing and model
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown",
        )),
        ("to_string", FunctionTransformer(
            lambda X: X.astype(str),
            feature_names_out="one-to-one",
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ]
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Probability that each test booking will be canceled
probabilities = model.predict_proba(X_test)[:, 1]

# Actual outcomes are used only to evaluate the policy,
# not to choose which bookings to contact.
actual = y_test.to_numpy()
n = len(actual)
total_cancellations = int(actual.sum())


# --------------------------------------------------
# 6. Evaluate fixed probability thresholds
# --------------------------------------------------

threshold_rows = []

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    selected = probabilities >= threshold

    tp = int(((selected) & (actual == 1)).sum())
    fp = int(((selected) & (actual == 0)).sum())
    fn = int(((~selected) & (actual == 1)).sum())

    threshold_rows.append({
        "policy": "Probability threshold",
        "threshold": threshold,
        "share_contacted_percent": round(100 * selected.mean(), 2),
        "bookings_contacted": int(selected.sum()),
        "true_cancellations_found": tp,
        "false_alerts": fp,
        "cancellations_missed": fn,
        "precision_percent": round(
            100 * tp / (tp + fp), 2
        ) if (tp + fp) else 0,
        "recall_percent": round(
            100 * tp / total_cancellations, 2
        ) if total_cancellations else 0,
    })


# --------------------------------------------------
# 7. Evaluate limited follow-up capacity
# --------------------------------------------------

ranking = np.argsort(-probabilities)

capacity_rows = []

for share in [0.10, 0.20, 0.30]:
    k = max(1, int(np.ceil(n * share)))

    selected_indices = ranking[:k]
    selected_actual = actual[selected_indices]

    found = int(selected_actual.sum())

    capacity_rows.append({
        "policy": "Top risk by capacity",
        "capacity_percent": int(share * 100),
        "bookings_in_test": n,
        "bookings_contacted": k,
        "actual_cancellations_found": found,
        "precision_percent": round(
            100 * found / k, 2
        ),
        "share_of_all_cancellations_found_percent": round(
            100 * found / total_cancellations, 2
        ) if total_cancellations else 0,
        "overall_test_cancellation_rate_percent": round(
            100 * total_cancellations / n, 2
        ) if n else 0,
    })


# --------------------------------------------------
# 8. Save reports
# --------------------------------------------------

threshold_path = RESULTS_DIR / "business_threshold_analysis.csv"
capacity_path = RESULTS_DIR / "business_followup_capacity.csv"

pd.DataFrame(threshold_rows).to_csv(
    threshold_path, index=False, encoding="utf-8-sig"
)

capacity_df = pd.DataFrame(capacity_rows)
capacity_df.to_csv(
    capacity_path, index=False, encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("PROBABILITY THRESHOLD RESULTS")
print("=" * 65)
print(pd.DataFrame(threshold_rows).to_string(index=False))

print("\n" + "=" * 65)
print("FOLLOW-UP CAPACITY RESULTS")
print("=" * 65)
print(capacity_df.to_string(index=False))

print("\nReports saved:")
print(threshold_path)
print(capacity_path)

print("\nImportant: these are retrospective estimates,")
print("not proof that contacting customers reduces cancellations.")