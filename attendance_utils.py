from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_dataset_path() -> Path:
    root = get_project_root()
    candidates = [
        root / "data" / "attendance.csv",
        root / "data" / "2018-2019_Daily_Attendance_20240429.csv",
        root / "2018-2019_Daily_Attendance_20240429.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Attendance dataset not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_attendance_data() -> pd.DataFrame:
    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required_columns = ["School DBN", "Date", "Enrolled", "Absent", "Present", "Released"]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    df = df.copy()
    df["School DBN"] = df["School DBN"].astype("string").str.strip()

    if "Date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(
                df["Date"].astype(str).str.zfill(8),
                format="%Y%m%d",
                errors="coerce",
            )
        else:
            df["Date"] = pd.to_datetime(
                df["Date"].astype(str).str.strip(),
                errors="coerce",
            )

    numeric_columns = ["Enrolled", "Absent", "Present", "Released"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_rows = df[required_columns].isna().any(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"Dataset contains {int(invalid_rows.sum())} rows with missing or invalid required values; "
            "no rows were silently discarded."
        )
    return df.reset_index(drop=True)


def build_attendance_chart_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    average_counts = (
        df[["Present", "Absent"]]
        .mean()
        .rename_axis("Attendance")
        .reset_index(name="Average students per school-day")
    )

    count_columns = ["Present", "Absent", "Enrolled"]
    daily = df.groupby("Date")[count_columns].sum().sort_index()
    daily["School-Day Records"] = df.groupby("Date").size()
    daily["Attendance Percentage"] = daily["Present"].div(daily["Enrolled"]) * 100
    daily["Average Present per School-Day"] = daily["Present"].div(daily["School-Day Records"])
    daily["Average Absent per School-Day"] = daily["Absent"].div(daily["School-Day Records"])
    daily = daily.reset_index()

    monthly = (
        df.assign(Month=df["Date"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)[count_columns]
        .sum()
    )
    monthly["Attendance Percentage"] = monthly["Present"].div(monthly["Enrolled"]) * 100

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.assign(Weekday=df["Date"].dt.day_name()).groupby("Weekday")[count_columns].sum()
    weekday["Attendance Percentage"] = weekday["Present"].div(weekday["Enrolled"]) * 100
    weekday = weekday.reindex(weekday_order).dropna(subset=["Present"]).reset_index()

    schools = df.groupby("School DBN")[count_columns].sum()
    schools["Attendance Percentage"] = schools["Present"].div(schools["Enrolled"]) * 100
    schools = schools.sort_values("Attendance Percentage", ascending=False).head(10).reset_index()

    return average_counts, daily, monthly, weekday, schools


def build_prediction_features(
    df: pd.DataFrame,
    school_codes: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    frame = df.copy()
    frame["Year"] = frame["Date"].dt.year
    frame["Month"] = frame["Date"].dt.month
    frame["Day"] = frame["Date"].dt.day
    frame["DayOfWeek"] = frame["Date"].dt.dayofweek
    if school_codes is None:
        school_names = sorted(frame["School DBN"].dropna().unique())
        school_codes = {name: index for index, name in enumerate(school_names)}
        frame["School_Code"] = frame["School DBN"].map(school_codes).fillna(-1).astype(int)
    else:
        frame["School_Code"] = frame["School DBN"].map(school_codes).fillna(-1).astype(int)

    feature_columns = [
        "School_Code",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
    ]

    X = frame[feature_columns]
    y = frame["Present"] if "Present" in frame.columns else None
    return X, y
