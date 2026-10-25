from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "results" / "hotel_bookings_analysis_ready.csv"

st.set_page_config(
    page_title="LUXE | Hotel Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. Premium Dark Theme
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0A1020;
    --panel: #111B2E;
    --border: #24334B;
    --text: #F1F5FC;
    --muted: #91A2BC;
    --teal: #55D6BE;
    --blue: #639BFF;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #152743 0%, transparent 38%),
        #0A1020;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0D1729;
    border-right: 1px solid #24334B;
}

[data-testid="stSidebar"] * {
    color: #E5ECF7;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label {
    color: var(--text);
}

.hero {
    padding: 27px 30px;
    border: 1px solid #2A405C;
    border-radius: 20px;
    background: linear-gradient(115deg, #162943, #111B2E 65%, #12343C);
    margin-bottom: 25px;
}

.eyebrow {
    color: #55D6BE;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(27px, 3vw, 40px);
    font-weight: 800;
    letter-spacing: -1.4px;
    margin: 9px 0;
    color: #F5F8FF;
}

.hero-subtitle {
    color: #A9B9D0;
    font-size: 14px;
    line-height: 1.7;
}

.section-label {
    color: #DDE8F7;
    font-size: 17px;
    font-weight: 700;
    margin: 15px 0 13px 0;
}

.kpi-card {
    background: linear-gradient(145deg, #15233A, #101A2B);
    border: 1px solid #273953;
    border-radius: 16px;
    padding: 20px 20px 18px 20px;
    min-height: 140px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.kpi-label {
    color: #9AADC7;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 13px;
}

.kpi-value {
    color: #F3F7FF;
    font-size: clamp(24px, 2.2vw, 34px);
    font-weight: 800;
    letter-spacing: -1px;
}

.kpi-note {
    color: #55D6BE;
    font-size: 11px;
    margin-top: 10px;
}

.panel {
    background: #111B2E;
    border: 1px solid #25354D;
    border-radius: 16px;
    padding: 19px 20px;
    margin-bottom: 12px;
}

.panel-title {
    font-size: 16px;
    font-weight: 700;
    color: #F0F5FD;
    margin-bottom: 3px;
}

.panel-subtitle {
    font-size: 11px;
    color: #91A2BC;
    margin-bottom: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293B55;
    border-radius: 10px;
    overflow: hidden;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #15233A;
    border-color: #2B405D;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #91A2BC;
}

hr {
    border-color: #263750;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Load Data
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

required_columns = {"hotel", "is_canceled"}

if not required_columns.issubset(df.columns):
    st.error("The dataset is missing required columns.")
    st.stop()

# --------------------------------------------------
# 4. Sidebar Filters
# --------------------------------------------------

st.sidebar.markdown("## HOTEL INTELLIGENCE")
st.sidebar.caption("ANALYTICS CONTROL CENTER")
st.sidebar.divider()

st.sidebar.markdown("### Filters")

hotel_options = ["All Hotels"] + sorted(df["hotel"].dropna().unique().tolist()
)

selected_hotel = st.sidebar.selectbox(
    "Hotel Type",
    hotel_options,
)

month_column = "arrival_date_month"
month_options = ["All Months"]

if month_column in df.columns:
    month_options += sorted(
        df[month_column].dropna().unique().tolist(),
        key=lambda month: (
            pd.to_datetime(month, format="%B").month
            if month in [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ]
            else 13
        )
    )

selected_month = st.sidebar.selectbox(
    "Arrival Month",
    month_options,
)

st.sidebar.divider()
st.sidebar.caption("LUXE ANALYTICS • HOTEL PERFORMANCE")

# --------------------------------------------------
# 5. Apply Filters
# --------------------------------------------------

filtered_df = df.copy()

if selected_hotel != "All Hotels":
    filtered_df = filtered_df[
        filtered_df["hotel"] == selected_hotel
    ]

if selected_month != "All Months" and month_column in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df[month_column] == selected_month
    ]

if filtered_df.empty:
    st.warning("No bookings match the selected filters.")
    st.stop()

total_bookings = len(filtered_df)
canceled_bookings = int(filtered_df["is_canceled"].sum())
not_canceled = total_bookings - canceled_bookings
cancel_rate = canceled_bookings / total_bookings * 100

previous_rate = (
    df["is_canceled"].mean() * 100
    if len(df) > 0 else 0
)

# --------------------------------------------------
# 6. Hero Header
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">HOSPITALITY DATA INTELLIGENCE</div>
    <div class="hero-title">Hotel Booking Analytics</div>
    <div class="hero-subtitle">
        Explore booking performance, cancellation patterns,
        and key hospitality metrics in one place.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 7. KPI Cards
# --------------------------------------------------

st.markdown(
    '<div class="section-label">Performance Overview</div>',
    unsafe_allow_html=True
)

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">TOTAL BOOKINGS</div>
        <div class="kpi-value">{total_bookings:,}</div>
        <div class="kpi-note">Filtered reservations</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">CANCELED BOOKINGS</div>
        <div class="kpi-value">{canceled_bookings:,}</div>
        <div class="kpi-note">Reservations canceled</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">CANCELLATION RATE</div>
        <div class="kpi-value">{cancel_rate:.2f}%</div>
        <div class="kpi-note">Overall dataset: {previous_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 8. Charts
# --------------------------------------------------

st.markdown(
    '<div class="section-label">Booking Insights</div>',
    unsafe_allow_html=True
)

chart_col, insight_col = st.columns([1.65, 1])

with chart_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Booking Status Distribution</div>
        <div class="panel-subtitle">
            Canceled versus successfully retained bookings
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#111B2E")
    ax.set_facecolor("#111B2E")

    labels = ["Canceled", "Not Canceled"]
    values = [canceled_bookings, not_canceled]
    colors = ["#639BFF", "#55D6BE"]

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.48,
        zorder=3,
    )

    ax.set_ylabel(
        "Number of Bookings",
        color="#A7B7CD",
        fontsize=9,
    )
    ax.tick_params(colors="#B8C7DA", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#33445D")
    ax.spines["bottom"].set_color("#33445D")
    ax.grid(axis="y", color="#293A52", alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color="#F1F5FC",
            fontsize=9,
            fontweight="bold",
        )

    ax.margins(y=0.18)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with insight_col:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Cancellation Snapshot</div>
        <div class="panel-subtitle">Current filtered selection</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="panel">
        <div style="color:#91A2BC;font-size:12px;">
            CANCELLATION SHARE
        </div>
        <div style="font-size:32px;font-weight:800;color:#639BFF;
                    margin:9px 0;">
            {cancel_rate:.2f}%
        </div>
        <div style="height:8px;background:#25354D;border-radius:8px;">
            <div style="height:8px;width:{min(cancel_rate, 100):.2f}%;
                        background:linear-gradient(90deg,#639BFF,#55D6BE);
                        border-radius:8px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    margin-top:15px;font-size:12px;color:#A7B7CD;">
            <span>Canceled</span>
            <b style="color:#F1F5FC;">{canceled_bookings:,}</b>
        </div>
        <div style="display:flex;justify-content:space-between;
                    margin-top:10px;font-size:12px;color:#A7B7CD;">
            <span>Not Canceled</span>
            <b style="color:#55D6BE;">{not_canceled:,}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# 9. Compact Data Table
# --------------------------------------------------

st.markdown(
    '<div class="section-label">Reservation Explorer</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="panel">
    <div class="panel-title">Booking Data Preview</div>
    <div class="panel-subtitle">
        A compact view of the first 10 filtered reservations
    </div>
</div>
""", unsafe_allow_html=True)

preferred_columns = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_month",
    "adults",
    "children",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
]

visible_columns = [
    col for col in preferred_columns
    if col in filtered_df.columns
]

preview = filtered_df[visible_columns].head(10).copy()

if "is_canceled" in preview.columns:
    preview["is_canceled"] = preview["is_canceled"].map(
        {1: "Canceled", 0: "Not Canceled"}
    )

st.dataframe(
    preview,
    use_container_width=True,
    height=260,
    hide_index=True,
)

st.caption(
    f"Showing {len(preview)} of {total_bookings:,} filtered bookings."
)

# --------------------------------------------------
# 10. Footer
# --------------------------------------------------

st.divider()
st.markdown(
    "<div style='text-align:center;color:#71839E;font-size:11px;'>"
    "LUXE HOTEL INTELLIGENCE  •  POWERED BY PYTHON & STREAMLIT"
    "</div>",
    unsafe_allow_html=True,
)