from pathlib import Path
import pandas as pd

# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Data" / "hotel_bookings.csv"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Early prediction features
# ==========================================

FEATURES = [
    "hotel",
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
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
]


# ==========================================
# 3. Timing audit definitions
# These are preliminary assessments, not proof.
# ==========================================

AUDIT = {
    "hotel": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Hotel type is normally known when a booking is created.",
        "recommendation": "Keep provisionally",
    },
    "lead_time": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Can be calculated from booking date and arrival date.",
        "recommendation": "Keep provisionally",
    },
    "stays_in_weekend_nights": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Planned weekend stay length is normally provided at booking.",
        "recommendation": "Keep provisionally",
    },
    "stays_in_week_nights": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Planned weekday stay length is normally provided at booking.",
        "recommendation": "Keep provisionally",
    },
    "adults": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Number of adults is normally supplied with the reservation.",
        "recommendation": "Keep provisionally",
    },
    "children": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Confirm when child counts are entered or updated.",
        "recommendation": "Verify booking-time availability",
    },
    "babies": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Baby count is normally part of guest details.",
        "recommendation": "Keep provisionally",
    },
    "is_repeated_guest": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Requires a reliable customer history available at prediction time.",
        "recommendation": "Verify historical calculation",
    },
    "previous_cancellations": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Must include only cancellations before the current booking.",
        "recommendation": "Verify historical calculation",
    },
    "previous_bookings_not_canceled": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Must count only earlier bookings known at prediction time.",
        "recommendation": "Verify historical calculation",
    },
    "adr": {
        "assessment": "Verify",
        "risk": "High",
        "reason": "The dataset's rate may not exactly represent the rate known at initial booking.",
        "recommendation": "Verify timing or exclude in a strict early model",
    },
    "meal": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Meal plan is generally selected during booking.",
        "recommendation": "Keep provisionally",
    },
    "country": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "May be missing or populated later; confirm the booking workflow.",
        "recommendation": "Verify booking-time availability",
    },
    "market_segment": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Usually associated with the booking source; confirm when recorded.",
        "recommendation": "Verify booking-time availability",
    },
    "distribution_channel": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Usually associated with the booking channel; confirm when recorded.",
        "recommendation": "Verify booking-time availability",
    },
    "reserved_room_type": {
        "assessment": "Likely available",
        "risk": "Low",
        "reason": "Reserved room type is generally known when booking.",
        "recommendation": "Keep provisionally",
    },
    "deposit_type": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Confirm whether the deposit policy is known and stable at prediction time.",
        "recommendation": "Verify booking-time availability",
    },
    "customer_type": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Confirm whether the customer classification is assigned at booking.",
        "recommendation": "Verify booking-time availability",
    },
    "agent": {
        "assessment": "Verify",
        "risk": "Medium",
        "reason": "Agent ID may be absent or assigned after a booking is created.",
        "recommendation": "Verify booking-time availability",
    },
}


# ==========================================
# 4. Audit function
# ==========================================

def main():
    print("=" * 75)
    print("EARLY PREDICTION FEATURE AVAILABILITY AUDIT")
    print("=" * 75)

    print(f"\nData file:\n{DATA_FILE}")

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check the Data folder and the dataset filename."
        )

    # Read the dataset for column and quality checks.
    df = pd.read_csv(DATA_FILE, low_memory=False)

    rows = []

    for feature in FEATURES:
        exists = feature in df.columns
        info = AUDIT[feature]

        if exists:
            missing_count = int(df[feature].isna().sum())
            missing_percent = round(
                df[feature].isna().mean() * 100, 2
            )
            unique_count = int(df[feature].nunique(dropna=True))

            examples = (
                df[feature]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )
            examples = " | ".join(examples)
        else:
            missing_count = None
            missing_percent = None
            unique_count = None
            examples = ""

        rows.append({
            "feature": feature,
            "exists_in_dataset": exists,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_values": unique_count,
            "sample_values": examples,
            "initial_timing_assessment": info["assessment"],
            "timing_risk": info["risk"],
            "assessment_reason": info["reason"],
            "preliminary_recommendation": info["recommendation"],
            "timing_verified_from_source": False,
        })

    report = pd.DataFrame(rows)

    # ------------------------------------------
    # Save detailed report
    # ------------------------------------------

    output_file = RESULTS_DIR / "feature_availability_audit.csv"
    report.to_csv(output_file, index=False, encoding="utf-8-sig")

    # ------------------------------------------
    # Print report and summary
    # ------------------------------------------

    print("\nFEATURE AUDIT SUMMARY")
    print("-" * 75)
    print(f"Dataset rows: {len(df):,}")
    print(f"Features requested: {len(FEATURES)}")
    print(
        "Features found: "
        f"{int(report['exists_in_dataset'].sum())}"
    )
    print(
        "Features missing from dataset: "
        f"{int((~report['exists_in_dataset']).sum())}"
    )
    print(
        "Preliminary low-risk features: "
        f"{int((report['timing_risk'] == 'Low').sum())}"
    )
    print(
        "Features requiringverification: "
        f"{int((report['initial_timing_assessment'] == 'Verify').sum())}"
    )
    print(
        "High timing-risk features: "
        f"{int((report['timing_risk'] == 'High').sum())}"
    )

    print("\nDETAILED REPORT")
    print("-" * 75)

    print(
        report[
            [
                "feature",
                "exists_in_dataset",
                "missing_percent",
                "initial_timing_assessment",
                "timing_risk",
                "preliminary_recommendation",
            ]
        ].to_string(index=False)
    )

    print("\nIMPORTANT LIMITATION")
    print(
        "Column presence and missing-value checks cannot prove when "
        "a value became available. Timing assessments are preliminary "
        "and must be checked against dataset documentation or source "
        "system records."
    )

    print(f"\nReport saved to:\n{output_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()