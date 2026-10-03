# Smart Attendance Management System

A professional attendance analytics and forecasting project built using Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Streamlit. The project analyzes daily school attendance records, visualizes trends, and evaluates machine learning models for attendance forecasting.

## 1. Project Overview

This system analyzes 277,153 daily public-school attendance records from 2018-2019, provides an interactive Streamlit dashboard and a command-line record manager, and compares reproducible attendance forecasting models. The project uses paths relative to its own directory and is structured for Streamlit Community Cloud deployment.

## 2. Problem Statement

Education institutions often need better ways to understand attendance trends across school buildings, days, and months. Manual record tracking is slow and error-prone. A data-driven dashboard helps reveal absentee patterns, daily fluctuations, and school-level performance using historical attendance data.

## 3. Objectives

- Analyze daily attendance records
- Detect attendance trends and anomalies
- Search and inspect school attendance records
- Evaluate multiple machine learning models
- Provide an interactive Streamlit dashboard
- Prepare deployable documentation and project structure

## 4. Features

- Attendance overview metrics
- School and date search
- Trend analysis across time
- Monthly and school-level summaries
- Leakage-aware model comparison and prediction
- Dataset exploration with date filters
- Automated chart and model-metric export to `reports/`
- Dark-mode dashboard UI

## 5. Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## 6. Dataset

The project uses the attendance dataset file:

- 2018-2019_Daily_Attendance_20240429.csv

This file contains daily attendance records with the following key columns:

- School DBN
- Date
- Enrolled
- Absent
- Present
- Released

Important note: the original date field is stored as a compact integer in the form `YYYYMMDD`, so the project converts it to a proper pandas datetime before analysis. This avoids the common Unix-epoch issue that can otherwise show values like `1970-01-01`.

## 7. Machine Learning Models

The project compares the following regressors:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNeighbors Regressor
- Gradient Boosting Regressor

Model metrics include:

- MAE
- RMSE
- R² Score

A critical data relationship is:

`Present = Enrolled - Absent - Released`

All same-day attendance counts (`Enrolled`, `Absent`, and `Released`) are excluded from predictors. Models use school identity and calendar fields only. Unique dates are split chronologically into training (first 70%), validation (next 15%), and final test (latest 15%) windows. A fixed-seed sample of up to 20,000 training rows and 5,000 rows per evaluation window keeps the five-model comparison practical and reproducible. The validation MAE selects the final estimator; reported MAE, RMSE, and R² use the later untouched test window.

## 8. Dashboard Features

The dashboard includes:

- Dashboard overview cards
- Average present/absent counts per school-day, weighted daily/monthly attendance rates, weekday rates, and top-school rates
- Search by school and date
- Model comparison table and MAE chart
- School/date attendance prediction
- Filterable dataset explorer and summary

Chart rates are calculated from grouped sums (`Present / Enrolled * 100`). Count comparisons use the average per school-day rather than summing every row into totals of more than 150 million. The original attendance totals remain available in the overview KPIs.

## 9. Project Structure

```text
Attendance-Management-System/
├── dashboard.py
├── ams.py
├── analysis.py
├── ML_models.py
├── prediction.py
├── visualization.py
├── attendance_utils.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
├── .gitignore
├── 2018-2019_Daily_Attendance_20240429.csv
├── reports/
│   └── generated charts and model metrics
└── tests/
    └── test_attendance_pipeline.py
```

## 10. Installation

Use Python 3.10 or newer. Python 3.14.4 was used for local verification. From the project folder, create and activate a virtual environment, then install dependencies:

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## 11. How to Run

Run the analysis script:

```bash
python analysis.py
```

Run the ML comparison script:

```bash
python ML_models.py
```

Run the prediction script:

```bash
python prediction.py
```

Run the CLI attendance manager:

```bash
python ams.py
```

## 12. How to Run Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Then open the local Streamlit URL shown in the terminal.

The dashboard entry point is `dashboard.py`. The dataset is included in the repository and is resolved relative to the project directory; no machine-specific paths or secrets are required.

## 13. Model Evaluation

Run `python ML_models.py` to print the holdout metrics for each regressor:

- MAE
- RMSE
- R² Score

The following results were measured with Python 3.14.4, scikit-learn 1.9.0, and the fixed random seed:

| Model | Validation MAE | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 17.59 | 39.24 | 83.55 | 0.9637 |
| Decision Tree | 18.29 | 40.90 | 84.56 | 0.9628 |
| Gradient Boosting | 202.76 | 209.26 | 313.08 | 0.4896 |
| KNN | 203.62 | 212.86 | 363.10 | 0.3136 |
| Linear Regression | 268.88 | 271.05 | 432.15 | 0.0277 |

Random Forest is selected by validation MAE and is the final model used by `prediction.py` and the dashboard. These are historical holdout estimates, not a guarantee of future performance.

The data file is approximately 8.1 MB, below GitHub's standard 100 MB per-file limit, so Git LFS is not needed.

## 14. Screenshots

The project can generate chart images in the `reports/` directory. Example files include:

- attendance_distribution.png
- attendance_trend.png
- school_attendance.png
- attendance_percentage.png
- model_comparison.png
- model_metrics.csv

Generate reports with:

```bash
python visualization.py
```

## 15. Streamlit Community Cloud Deployment

1. Push this repository to a GitHub repository you control.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository, branch, and `dashboard.py` as the main file path.
4. Deploy. The service installs packages from `requirements.txt`; the included dataset is read from the project directory.

No secrets are needed for the current application. Keep `.streamlit/secrets.toml` and `.env` out of Git.

## 16. Future Scope

- Add school-level predictive dashboards
- Introduce anomaly detection for irregular attendance
- Build an automated monthly reporting workflow
- Export findings to PDF or Excel
- Add user authentication for institutional deployment

## 17. Author

Smart Attendance Management System

---

This project is prepared for local execution and Streamlit deployment with portable relative file paths.
