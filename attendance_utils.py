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
