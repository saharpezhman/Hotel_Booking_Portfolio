from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# =====================================================
# 1. Load data
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

possible_paths = [
    BASE_DIR / "hotel_bookings_analysis_ready.csv",
    RESULTS_DIR / "hotel_bookings_analysis_ready.csv",
]

DATA_PATH = next(
    (path for path in possible_paths if path.exists()),
    None,
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "hotel_bookings_analysis_ready.csv was not found "
        "in the project folder or results folder."
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["is_canceled"]).copy()

y = df["is_canceled"].astype(int)

print("=" * 65)
print("COMPARISON OF CANCELLATION PREDICTION MODELS")
print("=" * 65)
print("Dataset:", df.shape)


# =====================================================
# 2. Define features
# =====================================================

# Original model: 24 features
original_numeric = [
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

original_categorical = [
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

# Early model: 19 features
early_numeric = [
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

early_categorical = [
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

all_features = list(
    dict.fromkeys(
        original_numeric
        + original_categorical
        + early_numeric
        + early_categorical
    )
)

missing_columns = [
    col for col in ["is_canceled"] + all_features
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

# One shared split for both models
X_train, X_test, y_train, y_test = train_test_split(
    df[all_features],
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# =====================================================
# 3. Create preprocessing and model pipeline
# =====================================================

def build_model(numeric_features, categorical_features,
                class_weight=None):

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
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


# =====================================================
# 4. Define the two models
# =====================================================

models = {
    "Original Random Forest": build_model(
        original_numeric,
        original_categorical,
        class_weight=None,
    ),
    "Early Random Forest": build_model(
        early_numeric,
        early_categorical,
        class_weight="balanced",
    ),
}


# =====================================================
# 5. Train and evaluate
# =====================================================

results = []
confusion_matrices = {}

for name, model in models.items():

    print(f"\nTraining: {name} ...")

    if name == "Original Random Forest":
        features = original_numeric + original_categorical
    else:
        features = early_numeric + early_categorical

    model.fit(X_train[features], y_train)

    predictions = model.predict(X_test[features])
    probabilities = model.predict_proba(X_test[features])[:, 1]

    metrics = {
        "Model": name,
        "Number of features": len(features),
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)

    from sklearn.metrics import confusion_matrix
    confusion_matrices[name] = confusion_matrix(
        y_test, predictions
    )

    print(f"\n{name}")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")


# =====================================================
# 6. Save comparison table
# =====================================================

results_df = pd.DataFrame(results)

csv_path = RESULTS_DIR / "cancellation_models_comparison.csv"
results_df.to_csv(csv_path, index=False)

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(results_df.round(4).to_string(index=False))


# =====================================================
# 7. Plot metric comparison
# =====================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
]

ax = results_df.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(12, 6),
)

ax.set_title("Cancellation Prediction Models Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

chart_path = RESULTS_DIR / "cancellation_models_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =====================================================
# 8. Plot confusion matrices
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, matrix) in zip(
    axes, confusion_matrices.items()
):
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not canceled", "Canceled"],
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()

matrix_path = RESULTS_DIR / "cancellation_models_confusion_matrices.png"
plt.savefig(matrix_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nFiles saved:")
print(csv_path)
print(chart_path)
print(matrix_path)

print("\nComparison completed successfully.")