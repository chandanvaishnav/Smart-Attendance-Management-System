from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attendance_utils import get_project_root, load_attendance_data


def save_chart(fig: plt.Figure, filename: str) -> None:
    output_dir = get_project_root() / "reports"
    output_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_attendance_data()

    total_present = int(df["Present"].sum())
    total_absent = int(df["Absent"].sum())

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(["Present", "Absent"], [total_present, total_absent], color=["#22d3ee", "#8b5cf6"])
    ax1.set_title("Total Present vs Absent Students")
    ax1.set_xlabel("Attendance Status")
    ax1.set_ylabel("Number of Students")
    save_chart(fig1, "attendance_distribution.png")

    daily_data = df.groupby("Date")["Present"].sum()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(daily_data.index, daily_data.values, color="#22d3ee", linewidth=2)
    ax2.set_title("Daily Attendance Trend")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Present Students")
    ax2.tick_params(axis="x", rotation=45)
    save_chart(fig2, "attendance_trend.png")

    top_schools = (
        df.groupby("School DBN")["Present"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.bar(top_schools.index, top_schools.values, color="#8b5cf6")
    ax3.set_title("Top 10 Schools by Average Attendance")
    ax3.set_xlabel("School DBN")
    ax3.set_ylabel("Average Present Students")
    ax3.tick_params(axis="x", rotation=45)
    save_chart(fig3, "school_attendance.png")

    daily_totals = df.groupby("Date")[["Present", "Absent"]].sum()
    daily_enrolled = df.groupby("Date")["Enrolled"].sum()
    daily_rate = daily_totals["Present"] / daily_enrolled * 100
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(daily_rate.index, daily_rate.values, color="#8b5cf6", linewidth=2)
    ax4.set_title("Daily Attendance Percentage")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Attendance Percentage")
    ax4.tick_params(axis="x", rotation=45)
    save_chart(fig4, "attendance_percentage.png")

    from ML_models import train_and_evaluate_models

    model_results, _, _, _ = train_and_evaluate_models()
    output_dir = get_project_root() / "reports"
    model_results.to_csv(output_dir / "model_metrics.csv", index=False)
    fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(13, 6))
    ax5.bar(model_results["Model"], model_results["MAE"], color="#22d3ee")
    ax5.set_title("Model MAE (lower is better)")
    ax5.tick_params(axis="x", rotation=35)
    ax6.bar(model_results["Model"], model_results["R2 Score"], color="#8b5cf6")
    ax6.set_title("Model R² (higher is better)")
    ax6.tick_params(axis="x", rotation=35)
    save_chart(fig5, "model_comparison.png")

    print("Reports saved to:", (get_project_root() / "reports").resolve())


if __name__ == "__main__":
    main()