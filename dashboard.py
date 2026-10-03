import streamlit as st
import pandas as pd

from ML_models import train_and_evaluate_models
from attendance_utils import load_attendance_data


st.set_page_config(page_title="Smart Attendance Dashboard", page_icon="🎓", layout="wide")


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_attendance_data()


@st.cache_resource
def get_model_registry_and_test_data():
    model_results, trained_models, X_test, y_test = train_and_evaluate_models()
    return model_results, trained_models, X_test, y_test


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #020817, #0b1120 40%, #111827); color: #f8fafc; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #020817, #111827); }
    .main-title {
        font-size: 2.5rem; font-weight: 800; letter-spacing: 0.04em;
        text-align: center; color: #f8fafc; margin-bottom: 0.25rem;
    }
    .subtitle {
        text-align: center; color: #94a3b8; font-size: 1.05rem; margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.5rem; color: #7dd3fc; font-weight: 700; margin: 1rem 0 0.75rem 0;
    }
    .metric-card {
        padding: 1.2rem 1rem; border-radius: 18px; background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid rgba(148, 163, 184, 0.25); box-shadow: 0 12px 30px rgba(15, 23, 42, 0.45);
        text-align: center; min-height: 120px; display: flex; flex-direction: column; justify-content: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #67e8f9; }
    .metric-label { font-size: 0.88rem; color: #cbd5e1; margin-top: 0.3rem; }
    .stDataFrame { background: rgba(15, 23, 42, 0.8); border-radius: 12px; }
    .stChart { border-radius: 12px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


df = get_dataset()

st.markdown('<div class="main-title">SMART ATTENDANCE MANAGEMENT SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data Analytics • Machine Learning • Forecasting</div>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.caption("Real-time attendance insights")
page = st.sidebar.radio(
    "Select a view",
    [
        "Dashboard Overview",
        "Attendance Analysis",
        "Search Records",
        "ML Prediction",
        "Model Comparison",
        "Visualizations",
        "Dataset Explorer",
    ],
)

if page == "Dashboard Overview":
    total_records = len(df)
    total_enrolled = int(df["Enrolled"].sum())
    total_present = int(df["Present"].sum())
    total_absent = int(df["Absent"].sum())
    attendance_rate = round((total_present / total_enrolled) * 100, 2) if total_enrolled else 0.0

    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (total_records, "Total Attendance Records"),
        (total_enrolled, "Total Enrolled Students"),
        (total_present, "Total Present"),
        (total_absent, "Total Absent"),
        (attendance_rate, "Attendance Rate"),
    ]
    for column, (value, label) in zip((col1, col2, col3, col4, col5), cards):
        with column:
            display_value = f"{value:,}" if isinstance(value, int) else f"{value:.2f}%"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{display_value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown('<div class="section-title">Recent Attendance Records</div>', unsafe_allow_html=True)
    st.dataframe(df.sort_values("Date", ascending=False).head(20), width="stretch")

elif page == "Attendance Analysis":
    st.markdown('<div class="section-title">Attendance Overview</div>', unsafe_allow_html=True)

    present_absent = pd.DataFrame(
        {"Students": [int(df["Present"].sum()), int(df["Absent"].sum())]},
        index=["Present", "Absent"],
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Present vs Absent")
        st.bar_chart(present_absent)
    with col2:
        st.subheader("Attendance Percentage")
        daily_rate = (df.groupby("Date")["Present"].sum() / df.groupby("Date")["Enrolled"].sum()) * 100
        st.line_chart(daily_rate.rename("Attendance %"))

    st.write("")
    st.subheader("Attendance Trend Over Time")
    daily = df.groupby("Date")["Present"].sum()
    st.line_chart(daily)

    st.write("")
    st.subheader("Monthly Attendance")
    monthly = df.assign(Month=df["Date"].dt.to_period("M").astype(str)).groupby("Month")["Present"].sum()
    st.bar_chart(monthly)

    st.write("")
    st.subheader("Top Schools by Average Attendance")
    school_summary = (
        (df.groupby("School DBN")["Present"].sum() / df.groupby("School DBN")["Enrolled"].sum())
        .sort_values(ascending=False)
        .head(10)
    )
    school_summary.name = "Attendance %"
    st.bar_chart(school_summary)

elif page == "Search Records":
    st.markdown('<div class="section-title">Search Records</div>', unsafe_allow_html=True)
    school_dbn = st.selectbox("School DBN", sorted(df["School DBN"].unique()))
    selected_date = st.selectbox("Date", sorted(df["Date"].dt.strftime("%Y-%m-%d").unique(), reverse=True))

    filtered = df[
        (df["School DBN"] == school_dbn) & (df["Date"].dt.strftime("%Y-%m-%d") == selected_date)
    ]

    st.success(f"Found {len(filtered)} matching records.")
    st.dataframe(filtered, width="stretch")

elif page == "Model Comparison":
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    model_results, trained_models, X_test, y_test = get_model_registry_and_test_data()
    st.dataframe(model_results, width="stretch")
    st.bar_chart(model_results.set_index("Model")["MAE"])
    st.caption("Metrics use a chronological holdout and school/calendar predictors only. Same-day attendance counts are excluded.")

elif page == "ML Prediction":
    st.markdown('<div class="section-title">Attendance Forecast</div>', unsafe_allow_html=True)
    model_results, trained_models, X_test, y_test = get_model_registry_and_test_data()
    best_model_name = model_results.iloc[0]["Model"]
    left, right = st.columns(2)
    with left:
        school = st.selectbox("School DBN", sorted(df["School DBN"].unique()))
    with right:
        forecast_date = st.date_input("Forecast date", value=(df["Date"].max() + pd.Timedelta(days=1)).date())

    if st.button("Generate prediction", type="primary"):
        from attendance_utils import build_prediction_features

        input_row = pd.DataFrame({"School DBN": [school], "Date": [pd.Timestamp(forecast_date)]})
        school_codes = {name: index for index, name in enumerate(sorted(df["School DBN"].unique()))}
        input_features, _ = build_prediction_features(input_row, school_codes)
        predicted = float(trained_models[best_model_name].predict(input_features)[0])
        st.success(f"{best_model_name} predicts {predicted:,.0f} students present.")
        st.metric("Predicted attendance", f"{predicted:,.0f} students")
        actual_rows = df[(df["School DBN"] == school) & (df["Date"].dt.date == forecast_date)]
        if not actual_rows.empty:
            actual = float(actual_rows["Present"].iloc[0])
            st.metric("Actual attendance", f"{actual:,.0f} students", f"Absolute error: {abs(actual - predicted):,.1f}")
        else:
            st.caption(f"Evaluation MAE for {best_model_name}: {model_results.iloc[0]['MAE']:.2f} students")
    st.caption("Forecast inputs are limited to school identity and calendar date; same-day enrollment and attendance counts are not used.")

elif page == "Visualizations":
    st.markdown('<div class="section-title">Attendance Visualizations</div>', unsafe_allow_html=True)
    daily_totals = df.groupby("Date")[["Present", "Absent"]].sum()
    daily_enrolled = df.groupby("Date")["Enrolled"].sum()
    daily_rate = (daily_totals["Present"] / daily_enrolled) * 100
    first, second = st.columns(2)
    with first:
        st.subheader("Present and Absent by Day")
        st.area_chart(daily_totals)
    with second:
        st.subheader("Daily Attendance Rate")
        st.line_chart(daily_rate.rename("Attendance %"))

elif page == "Dataset Explorer":
    st.markdown('<div class="section-title">Dataset Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary")
        summary = pd.DataFrame(
            {
                "Metric": ["Rows", "Columns", "Missing Values", "Date Start", "Date End"],
                "Value": [
                    str(len(df)),
                    str(len(df.columns)),
                    str(int(df.isnull().sum().sum())),
                    df["Date"].min().strftime("%Y-%m-%d"),
                    df["Date"].max().strftime("%Y-%m-%d"),
                ],
            }
        )
        st.dataframe(summary, width="stretch")
    with col2:
        st.subheader("Records")
        date_bounds = st.date_input(
            "Date range",
            value=(df["Date"].min().date(), df["Date"].max().date()),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
        )
        if len(date_bounds) == 2:
            start_date, end_date = date_bounds
            preview = df[df["Date"].dt.date.between(start_date, end_date)]
        else:
            preview = df
        st.dataframe(preview.head(100), width="stretch")

st.markdown("---")
st.caption("Smart Attendance Management System • Dashboard prepared for production and deployment.")