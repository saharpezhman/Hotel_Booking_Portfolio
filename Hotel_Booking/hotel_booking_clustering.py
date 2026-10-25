from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =====================================================
# 1. Load dataset
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
        "hotel_bookings_analysis_ready.csv not found."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("HOTEL BOOKING CLUSTERING WITH K-MEANS")
print("=" * 65)
print("Dataset shape:", df.shape)


# =====================================================
# 2. Select clustering features
# =====================================================

features = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "adr",
    "total_of_special_requests",
]

missing_columns = [
    col for col in features if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

X = df[features].copy()

# Ensure all selected columns are numeric
for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Replace infinite values with missing values
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Fill missing values using median
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Standardize features so larger-scale columns do not dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Features used:", features)
print("Rows used for clustering:", X_scaled.shape[0])


# =====================================================
# 3. Find a suitable number of clusters
# =====================================================

inertias = []
silhouette_scores = []
k_values = range(2, 8)

# A reproducible sample makes silhouette calculation faster
sample_size = min(5000, len(X_scaled))

for k in k_values:
    print(f"Testing k = {k} ...")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)

    # Use the same fixed sample for silhouette comparisons
    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=sample_size,
        random_state=42,
    )

    silhouette_scores.append(score)

    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Silhouette: {score:.4f}")


# Save evaluation results
evaluation_df = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertias,
    "silhouette_score": silhouette_scores,
})

evaluation_path = (
    RESULTS_DIR / "clustering_k_evaluation.csv"
)
evaluation_df.to_csv(evaluation_path, index=False)

best_index = silhouette_scores.index(max(silhouette_scores))
best_k = list(k_values)[best_index]

print("\n" + "=" * 65)
print("CLUSTER COUNT EVALUATION")
print("=" * 65)
print(evaluation_df.round(4).to_string(index=False))
print("\nBest k by silhouette score:", best_k)


# =====================================================
# 4. Plot Elbow curve
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.tight_layout()

elbow_path = RESULTS_DIR / "clustering_elbow.png"
plt.savefig(elbow_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# =====================================================
# 5. Plot Silhouette scores
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o",
)
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.tight_layout()

silhouette_path = (
    RESULTS_DIR / "clustering_silhouette.png"
)
plt.savefig(
    silhouette_path,
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


# =====================================================
# 6. Train final K-Means model
# =====================================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
)

cluster_labels = final_model.fit_predict(X_scaled)

# Add cluster labels to the original dataset
df["cluster"] = cluster_labels

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("cluster")
    .reset_index(name="booking_count")
)

cluster_counts["percentage"] = (
    cluster_counts["booking_count"]
    / len(df)
    * 100
)

print("\n" + "=" * 65)
print("CLUSTER SIZES")
print("=" * 65)
print(cluster_counts.round(2).to_string(index=False))


# =====================================================
# 7. Describe each cluster
# =====================================================

profile = (
    df.groupby("cluster")[features]
    .mean()
    .round(2)
)

print("\n" + "=" * 65)
print("CLUSTER PROFILES: MEAN VALUES")
print("=" * 65)
print(profile.to_string())


# Add cancellation rate as an interpretation metric.
# It is NOT used to train K-Means.
if "is_canceled" in df.columns:
    cancellation_rate = (
        df.groupby("cluster")["is_canceled"]
        .mean()
        .mul(100)
        .round(2)
        .rename("cancellation_rate_percent")
    )

    profile = profile.join(cancellation_rate)

print("\nCluster profile with cancellation rates:")
print(profile.to_string())


# =====================================================
# 8. Save results
# =====================================================

clustered_path = (
    RESULTS_DIR / "hotel_bookings_clustered.csv"
)
profile_path = (
    RESULTS_DIR / "hotel_booking_cluster_profiles.csv"
)
counts_path = (
    RESULTS_DIR / "hotel_booking_cluster_sizes.csv"
)

df.to_csv(clustered_path, index=False)
profile.to_csv(profile_path)
cluster_counts.to_csv(counts_path, index=False)

print("\nFiles saved:")
print(evaluation_path)
print(elbow_path)
print(silhouette_path)
print(clustered_path)
print(profile_path)
print(counts_path)

print("\nClustering completed successfully.")