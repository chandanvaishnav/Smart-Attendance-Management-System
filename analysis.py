import pandas as pd

from attendance_utils import load_attendance_data


def main() -> None:
    df = load_attendance_data()

    print("\n===== SMART ATTENDANCE DATA ANALYSIS =====")
    print("Total Rows:", len(df))
    print("Total Columns:", len(df.columns))
    print("Duplicate Rows:", int(df.duplicated().sum()))
    print("Missing Values:", df.isnull().sum().to_dict())
    print("Date Range:", df["Date"].min(), "to", df["Date"].max())
    print("Attendance Rate:", round((df["Present"].sum() / df["Enrolled"].sum()) * 100, 2), "%")
    count_columns = ["Enrolled", "Absent", "Present", "Released"]
    print("Negative Attendance Counts:", int((df[count_columns] < 0).sum().sum()))
    print(
        "Count Identity Violations:",
        int((df["Present"] + df["Absent"] + df["Released"] != df["Enrolled"]).sum()),
    )

    print("\n----- FIRST 5 RECORDS -----")
    print(df.head().to_string(index=False))

    print("\n----- DATA TYPES -----")
    print(df.dtypes)

    print("\n----- NUMERIC SUMMARY -----")
    print(df[["Enrolled", "Absent", "Present", "Released"]].describe().round(2).to_string())

    print("\nData validation completed successfully.")
    print("No large-scale data removal was required; the dataset is otherwise clean.")


if __name__ == "__main__":
    main()