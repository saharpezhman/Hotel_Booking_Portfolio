from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# ==================================================
# 1. Load dataset
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
y = pd.to_numeric(df["is_canceled"], errors="coerce")

valid = y.isin([0, 1])
X = X.loc[valid].copy()
y = y.loc[valid].astype(int)

# ==================================================
# 2. Split data
# Keep the same test split as the original model
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ==================================================
# 3. Preprocessing
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

# ==================================================
# 4. Train model
# ==================================================

print("Training Random Forest...")

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

# ==================================================
# 5. Calculate permutation importance
# ==================================================

print("\nCalculating permutation importance...")
print("This may take a few minutes.")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
)

# ==================================================
# 6. Print results
# ==================================================

print("\n" + "=" *65)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 65)

print(importance_df.to_string(
    index=False,
    formatters={
        "importance_mean": "{:.5f}".format,
        "importance_std": "{:.5f}".format,
    },
))

# ==================================================
# 7. Save CSV and chart
# ==================================================

output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "cancellation_feature_importance.csv"

importance_df.to_csv(csv_path, index=False)

top_features = (
    importance_df.head(15)
    .sort_values("importance_mean")
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"],
    capsize=3,
)

plt.axvline(0, linewidth=1)
plt.xlabel("Decrease in ROC-AUC after permutation")
plt.ylabel("Feature")
plt.title("Top Features for Hotel Cancellation Prediction")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

chart_path = (
    output_dir / "cancellation_feature_importance.png"
)

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nCSV saved to: {csv_path}")
print(f"Chart saved to: {chart_path}")