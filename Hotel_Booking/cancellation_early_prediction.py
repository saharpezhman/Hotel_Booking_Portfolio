from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("EARLY HOTEL BOOKING CANCELLATION PREDICTION")
print("=" * 65)
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Target and early-available features
# --------------------------------------------------

target = "is_canceled"

# Features selected as potentially available at booking time.
# Verify their availability against the dataset's data dictionary.
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

missing_columns = [
    col for col in [target] + features
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

df = df.dropna(subset=[target]).copy()

X = df[features].copy()
y = df[target].astype(int)

print("Number of features:", len(features))
print("Target distribution:")
print(y.value_counts(normalize=True).rename("proportion"))


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            ),
        ),
        (
            "to_string",
            FunctionTransformer(
                lambda X: X.astype(str),
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
# 4. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 5. Build and train model
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

print("\nTraining early prediction model...")
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

results = {
    "Model": "Early Random Forest",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision":precision_score(
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
print("MODEL EVALUATION")
print("=" * 65)

for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric:12s}: {value:.4f}")
    else:
        print(f"{metric:12s}: {value}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not canceled", "Canceled"],
        zero_division=0,
    )
)

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results_df = pd.DataFrame([results])
output_path = RESULTS_DIR / "cancellation_early_prediction_results.csv"
results_df.to_csv(output_path, index=False)

print("\nResults saved to:")
print(output_path)
print("\nDone.")