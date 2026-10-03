import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import ams
import attendance_utils
from attendance_utils import build_attendance_chart_data, build_prediction_features, load_attendance_data


class AttendancePipelineTests(unittest.TestCase):
    def test_dataset_dates_and_quality(self):
        df = load_attendance_data()

        self.assertEqual(len(df), 277153)
        self.assertEqual(
            list(df.columns),
            ["School DBN", "Date", "Enrolled", "Absent", "Present", "Released"],
        )
        self.assertTrue(df["Date"].notna().all())
        self.assertEqual(df["Date"].min().strftime("%Y-%m-%d"), "2018-09-04")
        self.assertEqual(df["Date"].max().strftime("%Y-%m-%d"), "2019-06-26")
        self.assertEqual(int(df.duplicated().sum()), 0)
        self.assertEqual(int(df.isna().sum().sum()), 0)

    def test_features_exclude_same_day_attendance_counts(self):
        df = load_attendance_data()
        features, target = build_prediction_features(df)

        self.assertNotIn("Enrolled", features.columns)
        self.assertNotIn("Absent", features.columns)
        self.assertNotIn("Released", features.columns)
        self.assertEqual(len(features), len(target))
        self.assertTrue((df["Present"] + df["Absent"] + df["Released"] == df["Enrolled"]).all())

    def test_chart_aggregations_are_realistic_and_conserve_totals(self):
        df = load_attendance_data()
        average_counts, daily, monthly, weekdays, schools = build_attendance_chart_data(df)

        average_by_status = average_counts.set_index("Attendance")["Average students per school-day"]
        self.assertAlmostEqual(average_by_status["Present"], df["Present"].mean(), places=6)
        self.assertAlmostEqual(average_by_status["Absent"], df["Absent"].mean(), places=6)
        self.assertLess(average_counts["Average students per school-day"].max(), 600)

        for grouped in (daily, monthly):
            self.assertEqual(int(grouped["Present"].sum()), int(df["Present"].sum()))
            self.assertEqual(int(grouped["Absent"].sum()), int(df["Absent"].sum()))
            self.assertTrue(grouped["Attendance Percentage"].between(0, 100).all())

        self.assertEqual(len(monthly), 10)
        self.assertEqual(monthly["Month"].tolist(), sorted(monthly["Month"].tolist()))
        self.assertEqual(daily["Date"].min().strftime("%Y-%m-%d"), "2018-09-04")
        self.assertEqual(daily["Date"].max().strftime("%Y-%m-%d"), "2019-06-26")
        self.assertTrue(weekdays["Attendance Percentage"].between(0, 100).all())
        self.assertEqual(len(schools), 10)
        self.assertTrue(schools["Attendance Percentage"].between(0, 100).all())
        self.assertTrue((average_counts["Average students per school-day"] >= 0).all())

    def test_invalid_required_values_are_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "invalid.csv"
            dataset.write_text(
                "School DBN,Date,Enrolled,Absent,Present,Released\n"
                "01M015,20180905,100,10,89,1\n"
                "01M015,not-a-date,100,10,89,1\n",
                encoding="utf-8",
            )
            with patch.object(attendance_utils, "resolve_dataset_path", return_value=dataset):
                with self.assertRaisesRegex(ValueError, "1 rows"):
                    load_attendance_data()

    def test_cli_menu_and_record_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "attendance.csv"
            with dataset.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["School DBN", "Date", "Enrolled", "Absent", "Present", "Released"])
                writer.writerow(["01M015", "20180905", 100, 10, 89, 1])

            answers = [
                "1", "", "2", "01M015", "3", "2018-09-05", "4", "01M015",
                "5", "6", "7", "8", "9", "02M111", "2018-09-06", "100", "5", "94", "1", "10",
            ]
            output = io.StringIO()
            with patch.object(ams, "FILE_NAME", dataset), patch("builtins.input", side_effect=answers):
                with redirect_stdout(output):
                    ams.main()

            text = output.getvalue()
            self.assertIn("Total Present Students: 89", text)
            self.assertIn("Total Absent Students: 10", text)
            self.assertIn("Attendance Percentage: 89.0 %", text)
            self.assertIn("New attendance record added successfully!", text)
            self.assertIn("Thank you for using the system!", text)
            saved = dataset.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(saved), 3)
            self.assertIn("02M111,20180906,100,5,94,1", saved)


if __name__ == "__main__":
    unittest.main()
