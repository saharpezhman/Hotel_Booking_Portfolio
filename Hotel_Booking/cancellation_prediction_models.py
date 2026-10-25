from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. Load the prepared dataset
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

possible_files = [
    BASE_DIR / "hotel_bookings_analysis_ready.csv",
    BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv",
]

DATA_FILE = next(
    (p for p in possible_files if p.exists()),
    None
)

if DATA_FILE is None:
    raise FileNotFoundError(
        "Could not find hotel_bookings_analysis_ready.csv"
    )

df = pd.read_csv(DATA_FILE)

print("=" * 65)
print("HOTEL BOOKING CANCELLATION PREDICTION")
print("=" * 65)
print(f"Dataset shape: {df.shape}")

# ==================================================
# 2. Select features available before the outcome
# ==================================================

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
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

categorical_features = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "agent",
]

# Keep only columns that actually exist in the dataset.
numeric_features = [
    col for col in numeric_features if col in df.columns
]

categorical_features = [
    col for col in categorical_features if col in df.columns
]

if "is_canceled" not in df.columns:
    raise ValueError("Target column 'is_canceled' not found.")

if not numeric_features and not categorical_features:
    raise ValueError("No usable feature columns were found.")

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()
y = pd.to_numeric(df["is_canceled"], errors="coerce")

# Remove rows with invalid target values.
valid_target = y.isin([0, 1])
X = X.loc[valid_target].copy()
y = y.loc[valid_target].astype(int)

print(f"Usable rows: {len(X):,}")
print(f"Number of features: {len(feature_columns)}")
print(f"Cancellation rate: {y.mean() * 100:.2f}%")

print("\nFeatures used:")
print(feature_columns)

# ==================================================
# 3. Preprocess missing and categorical values
# ==================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# ==================================================
# 4. Split data into training and test sets
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

# ==================================================
# 5. Define models
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

# ==================================================
# 6. Train and evaluate
# ==================================================

results = []
output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

for model_name, classifier in models.items():

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not Canceled", "Canceled"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    })

# ==================================================
# 7. Save model comparison
# ==================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="roc_auc",
    ascending=False,
)

results_file = output_dir / "cancellation_model_comparison.csv"

results_df.to_csv(results_file, index=False)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(results_df.to_string(index=False))

print(f"\nComparison saved to: {results_file}")