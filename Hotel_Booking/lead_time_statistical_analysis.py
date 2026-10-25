from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# -----------------------------------------
# 1. Load prepared dataset
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

candidates = [
    BASE_DIR / "hotel_bookings_analysis_ready.csv",
    BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv",
]

DATA_FILE = next(
    (p for p in candidates if p.exists()),
    None
)

if DATA_FILE is None:
    raise FileNotFoundError(
        "hotel_bookings_analysis_ready.csv not found"
    )

df = pd.read_csv(DATA_FILE)

required = ["hotel", "lead_time", "is_canceled"]

if not set(required).issubset(df.columns):
    raise ValueError(
        f"Required columns: {required}"
    )

df = df[required].copy()

df["lead_time"] = pd.to_numeric(
    df["lead_time"], errors="coerce"
)
df["is_canceled"] = pd.to_numeric(
    df["is_canceled"], errors="coerce"
)

df = df.dropna(subset=required)

df = df[
    df["lead_time"].ge(0)
    & df["is_canceled"].isin([0, 1])
].copy()

# -----------------------------------------
# 2. Create lead-time groups
# -----------------------------------------

bins = [-1, 7, 30, 90, 180, 365, np.inf]

labels = [
    "0-7 days",
    "8-30 days",
    "31-90 days",
    "91-180 days",
    "181-365 days",
    "Over 365 days",
]

df["lead_time_group"] = pd.cut(
    df["lead_time"],
    bins=bins,
    labels=labels
)

# -----------------------------------------
# 3. Statistical tests
# -----------------------------------------

def cramers_v(table, chi2):
    n = table.to_numpy().sum()
    r, k = table.shape

    if n == 0 or min(r - 1, k - 1) <= 0:
        return np.nan

    return np.sqrt(
        chi2 / (n * min(r - 1, k - 1))
    )

results = []

for hotel in ["City Hotel", "Resort Hotel"]:

    part = df[df["hotel"] == hotel]

    table = pd.crosstab(
        part["lead_time_group"],
        part["is_canceled"]
    ).reindex(
        index=labels,
        columns=[0, 1],
        fill_value=0
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(table)
    )

    v = cramers_v(table, chi2)

    print("\n" + "=" * 60)
    print(f"Hotel: {hotel}")
    print("=" * 60)

    print("\nContingency table:")
    print(
        table.rename(
            columns={
                0: "Not Canceled",
                1: "Canceled"
            }
        )
    )

    print(f"\nChi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.8g}")
    print(f"Cramer's V: {v:.4f}")
    print(
        "Expected cell count below 5:",
        bool((expected < 5).any())
    )

    if p_value < 0.05:
        print(
            "Conclusion: There is statistical "
            "evidence of an association."
        )
    else:
        print(
            "Conclusion: Evidence of an association "
            "was not sufficient at alpha=0.05."
        )

    results.append({
        "hotel": hotel,
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "cramers_v": v,
    })

# -----------------------------------------
# 4. Save results
# -----------------------------------------

results_df = pd.DataFrame(results)

output_dir = BASE_DIR / "results"
output_dir.mkdir(exist_ok=True)

output_file = (
    output_dir / "lead_time_statistical_results.csv"
)

results_df.to_csv(output_file, index=False)

print("\nStatistical results saved to:")
print(output_file)