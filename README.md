# 🛡️ Transaction Monitoring System (Rule-Based)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/rojuadeyemi/transaction-monitoring)](https://github.com/rojuadeyemi/transaction-monitoring/issues)

------------------------------------------------------------------------

## 📌 Overview

The **Transaction Monitoring System (TMS)** is a fully rule-based
solution designed to detect suspicious or high-risk financial
transactions.

The system applies predefined monitoring rules to transactional data,
assigns risk scores, and generates structured outputs for operational
review and reporting.

A dashboard is integrated for visualization, and the entire process is
automated to run daily.

------------------------------------------------------------------------

## 🎯 Objectives

-   Detect potentially fraudulent or risky transactions\
-   Apply configurable monitoring rules\
-   Generate weighted risk scores\
-   Automatically produce daily monitoring reports\
-   Provide dashboard-based visualization for stakeholders

------------------------------------------------------------------------

## 🔄 Automated Workflow

The system runs on a fully automated daily schedule:

1.  Windows Task Scheduler triggers a batch script\
2.  The batch script executes the transaction monitoring pipeline\
3.  Outputs are saved to a synchronized OneDrive
    folder\
4.  Power BI detects the updated dataset through Power Automate trigger\ 
5.  Power BI automatically refreshes the dashboard

This ensures continuous monitoring without manual intervention.

------------------------------------------------------------------------

## 🏗️ System Architecture

    Data Source (Transactions)
            ↓
    Data Cleaning & Transformation
            ↓
    Rule-Based Detection Engine
            ↓
    Risk Scoring
            ↓
    Flagged Transactions Output
            ↓
    OneDrive Sync
            ↓
    Power BI Auto Refresh Dashboard

------------------------------------------------------------------------

## 🔍 Monitoring Rules (Examples)

1.  **Large Transaction Rule** -- Flags transactions above a defined
    threshold\
2.  **High Frequency Rule** -- Flags multiple transactions within a
    short time window\
3.  **Velocity Rule** -- Flags rapid inflow and outflow movement\
4.  **Dormant Account Reactivation Rule** -- Flags large transactions
    after prolonged inactivity

All rules are configurable based on institutional risk appetite.

------------------------------------------------------------------------

## 🧮 Risk Scoring Methodology

Each triggered rule contributes a weighted score. Thus

Risk Score = Σ (Triggered Rule × Weight)

------------------------------------------------------------------------

## 📊 Dashboard & Reporting

-   Interactive dashboard built in Power BI\
-   Daily automated refresh\
-   Risk distribution analysis\
-   High-risk transaction drill-down\
-   Trend monitoring over time

------------------------------------------------------------------------
### Technologies Used:
* Python Frameworks and Libraries: Pandas, NumPy
* SQL (for data extraction)
* Batch Script for automating
* Power BI / Excel for reporting and risk visualization
* Windows Scheduler for scheduling
* Power Automating Power BI refresh
* Version Control: Git
------------------------------------------------------------------------

## 🚀 Installation

### Prerequisites

- Python 3.10-3.12
- Pip (Python package installer)


## 🚀 Project Structure

- 📂 **utility/** : package folder containing helper functions (e.g., data preprocessing, risk scoring, risk engine etc).
- 📄 **requirements.txt** : list of all required dependencies.

---

## 🔧 Setup Instructions

1. **Clone the repository:**

    ```sh
    git clone https://github.com/rojuadeyemi/diabetes-test-app.git
    cd diabetes-test-app
    ```

2. **Create a Virtual Environment**

For **Linux/Mac**:

```sh
python -m venv .venv
source .venv/bin/activate
```

For **Windows**:

```sh
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

Once the environment is active, install dependencies:

```sh
python.exe -m pip install -U pip
pip install -r requirements.txt
```
------------------------------------------------------------------------

## ▶️ Usage

To run manually:

``` bash
python utility/main.py
```

To run automatically:

-   Configure Windows Task Scheduler\
-   Point it to `/run_script.bat`\
-   Schedule daily execution
------------------------------------------------------------------------

## 🔐 Compliance Considerations

This solution aligns with risk-based transaction monitoring principles
and can support AML and internal fraud detection frameworks.

------------------------------------------------------------------------

## 🔄 Future Enhancements

-   Expand rule coverage\
-   Add alert notification system\
-   Integrate case management workflow\
-   Extend to near real-time monitoring

------------------------------------------------------------------------
