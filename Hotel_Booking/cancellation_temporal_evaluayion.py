from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# --------------------------------------------------
# Helper function: must be a top-level named function
# so that joblib can serialize the pipeline.
# --------------------------------------------------

def convert_to_string(X):
    return X.astype(str)


# --------------------------------------------------
# 1. Paths and data
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"
)

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "hotel_cancellation_model.pkl"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("TEMPORAL EVALUATION - HOTEL CANCELLATION")
print("=" * 65)
print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Build approximate booking date
# --------------------------------------------------

date_columns = [
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "lead_time",
]

missing = [col for col in date_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing date columns: {missing}")

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
    arrival_date
    - pd.to_timedelta(df["lead_time"], unit="D")
)

df = df.dropna(
    subset=["estimated_booking_date", "is_canceled"]
).copy()

df["is_canceled"] = df["is_canceled"].astype(int)

df = df.sort_values(
    "estimated_booking_date"
).reset_index(drop=True)


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

missing_features = [
    col for col in features if col not in df.columns
]

if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

X = df[features].copy()
y = df["is_canceled"].copy()


# --------------------------------------------------
# 4. Chronological train/test split by unique date
# --------------------------------------------------

unique_dates = sorted(
    df["estimated_booking_date"].dropna().unique()
)

if len(unique_dates) < 2:
    raise ValueError("Not enough unique booking dates.")

cutoff_index = int(len(unique_dates) * 0.80)
cutoff_index = max(
    1,
    min(cutoff_index, len(unique_dates) - 1),
)

cutoff_date = pd.Timestamp(unique_dates[cutoff_index])

train_mask = df["estimated_booking_date"] < cutoff_date
test_mask = df["estimated_booking_date"] >= cutoff_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

train_dates = df.loc[
    train_mask, "estimated_booking_date"
]
test_dates = df.loc[
    test_mask, "estimated_booking_date"
]

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing set is empty.")

print("\nTemporal cutoff date:", cutoff_date)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print(
    "Training date range:",
    train_dates.min(), "to", train_dates.max()
)
print(
    "Testing date range:",
    test_dates.min(), "to", test_dates.max()
)

print("\nTraining cancellation rate:", round(y_train.mean(), 4))
print("Testing cancellation rate:", round(y_test.mean(), 4))


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
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
                convert_to_string,
                feature_names_out="one-to-one",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]
)

print("\nTraining model on earlier bookings...")

model.fit(X_train, y_train)

print("\nSaving trained model...")

joblib.dump(model, MODEL_PATH)

print("Trained model saved to:")
print(MODEL_PATH)


# --------------------------------------------------
# 7. Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

results = {
    "Model": "Temporal Early Random Forest",
    "Train rows": len(X_train),
    "Test rows": len(X_test),
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
}

print("\n" + "=" * 65)
print("TEMPORAL MODEL RESULTS")
print("=" * 65)

for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric:15s}: {value:.4f}")
    else:
        print(f"{metric:15s}: {value}")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Not canceled", "Canceled"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 8. Save evaluation results
# --------------------------------------------------

output_path = (
    RESULTS_DIR / "cancellation_temporal_evaluation.csv"
)

pd.DataFrame([results]).to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nResults saved to:")
print(output_path)

print("\nDone.")