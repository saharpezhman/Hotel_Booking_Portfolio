from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

# ==================================================
# 1. Load data
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

candidates = [
    BASE_DIR / "hotel_bookings_analysis_ready.csv",
    BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv",
]

DATA_FILE = next(
    (p for p in candidates if p.exists()),
    None,
)

if DATA_FILE is None:
    raise FileNotFoundError(
        "hotel_bookings_analysis_ready.csv not found"
    )

df = pd.read_csv(DATA_FILE)

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

numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]

features = numeric_features + categorical_features

X = df[features].copy()
y = pd.to_numeric(
    df["is_canceled"], errors="coerce"
)

valid = y.isin([0, 1])
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# ==================================================
# 2. Split into train, validation and test sets
# ==================================================

X_train_val, X_test, y_train_val, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=y_train_val,
    )
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# ==================================================
# 3. Build preprocessing pipeline
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

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# ==================================================
# 4. Choose threshold using validation data only
# ==================================================

val_probabilities = model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_val,
        val_probabilities,
    )
)

# The last precision/recall point has no corresponding
# threshold, so use thresholds[:-1] for F1 calculation.
f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-12
    )
)

best_index= int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

print("\n" + "=" * 60)
print("THRESHOLD SELECTION ON VALIDATION SET")
print("=" * 60)
print(f"Default threshold: 0.50")
print(f"Selected threshold: {best_threshold:.4f}")
print(f"Validation F1 at selected threshold: {f1_values[best_index]:.4f}")

# ==================================================
# 5. Evaluate on the untouched test set
# ==================================================

test_probabilities = model.predict_proba(X_test)[:, 1]

predictions_default = (
    test_probabilities >= 0.50
).astype(int)

predictions_selected = (
    test_probabilities >= best_threshold
).astype(int)

def show_metrics(name, y_true, predictions, probabilities):
    print("\n" + "-" * 55)
    print(name)
    print("-" * 55)

    print(
        f"Precision: {precision_score(y_true, predictions, zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_true, predictions, zero_division=0):.4f}"
    )
    print(
        f"F1-score:  {f1_score(y_true, predictions, zero_division=0):.4f}"
    )
    print(
        f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}"
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Not Canceled", "Canceled"],
            zero_division=0,
        )
    )

show_metrics(
    "TEST RESULTS - DEFAULT THRESHOLD 0.50",
    y_test,
    predictions_default,
    test_probabilities,
)

show_metrics(
    "TEST RESULTS - VALIDATION-SELECTED THRESHOLD",
    y_test,
    predictions_selected,
    test_probabilities,
)

# ==================================================
# 6. Save results and precision-recall chart
# ==================================================

output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

comparison = pd.DataFrame([
    {
        "threshold": 0.50,
        "precision": precision_score(
            y_test, predictions_default, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions_default, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions_default, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, test_probabilities
        ),
    },
    {
        "threshold": best_threshold,
        "precision": precision_score(
            y_test, predictions_selected, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions_selected, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions_selected, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, test_probabilities
        ),
    },
])

comparison.to_csv(
    output_dir / "threshold_comparison.csv",
    index=False,
)

p, r, _ = precision_recall_curve(
    y_test, test_probabilities
)

plt.figure(figsize=(8, 6))
plt.plot(r, p)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Random Forest")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "random_forest_precision_recall.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nSaved threshold comparison and PR curve.")