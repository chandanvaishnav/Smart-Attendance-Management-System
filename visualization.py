from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attendance_utils import build_attendance_chart_data, get_project_root, load_attendance_data


def save_chart(fig: plt.Figure, filename: str) -> None:
    output_dir = get_project_root() / "reports"
    output_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_attendance_data()
    average_counts, daily, monthly, _, schools = build_attendance_chart_data(df)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(
        average_counts["Attendance"],
        average_counts["Average students per school-day"],
        color=["#22d3ee", "#8b5cf6"],
    )
    ax1.set_title("Average Present vs Absent per School-Day")
    ax1.set_xlabel("Attendance Status")
    ax1.set_ylabel("Average students per school-day")
    save_chart(fig1, "attendance_distribution.png")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(daily["Date"], daily["Attendance Percentage"], color="#22d3ee", linewidth=2)
    ax2.set_ylim(0, 100)
    ax2.set_title("Daily Attendance Rate")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Attendance percentage")
    ax2.tick_params(axis="x", rotation=45)
    save_chart(fig2, "attendance_trend.png")

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.bar(schools["School DBN"], schools["Attendance Percentage"], color="#8b5cf6")
    ax3.set_ylim(0, 100)
    ax3.set_title("Top 10 Schools by Attendance Rate")
    ax3.set_xlabel("School DBN")
    ax3.set_ylabel("Attendance percentage")
    ax3.tick_params(axis="x", rotation=45)
    save_chart(fig3, "school_attendance.png")

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.bar(monthly["Month"], monthly["Attendance Percentage"], color="#8b5cf6")
    ax4.set_ylim(0, 100)
    ax4.set_title("Monthly Attendance Rate")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Attendance percentage")
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