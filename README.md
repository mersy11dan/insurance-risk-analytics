# Insurance Risk Analytics

This repository contains an end-to-end insurance analytics workflow for the
AlphaCare Insurance Solutions (ACIS) risk analytics challenge.

The goal is to turn raw car-insurance data into evidence-backed business
recommendations for risk-based pricing, marketing, and claim-risk management.
The project combines exploratory data analysis, data version control,
hypothesis testing, predictive modeling, and a final business report.

## Project Overview

ACIS wants to identify low-risk customer, geographic, and vehicle segments so
the business can improve pricing decisions and target more profitable markets.

The analysis is designed to answer questions such as:

- Which provinces have higher or lower claim risk?
- Do men and women differ significantly in insurance risk?
- Do some zip codes produce better margins than others?
- Which vehicles or policy features are linked to high claims?
- Can claim severity be predicted well enough to support pricing decisions?
- Which model features are the strongest drivers of risk?

## Repository Structure

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                     # Local datasets tracked with DVC, not Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── reports/
│   └── final_report.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── tests/
│   ├── __init__.py
│   └── test_project_setup.py
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Optional development checks:

```bash
ruff check .
black .
pytest
```

## DVC

Raw and processed datasets should be stored under `data/` and tracked with DVC
instead of Git. This keeps the Git repository lightweight while preserving data
lineage and reproducibility.

Typical DVC commands:

```bash
dvc init
dvc add data/<dataset-file>
git add data/<dataset-file>.dvc .gitignore
git commit -m "Track insurance dataset with DVC"
```

Pipeline stages can be added to `dvc.yaml` as the project grows, for example
data preparation, feature engineering, model training, and evaluation.

## Notebooks

The notebooks are organized by project phase:

- `notebooks/01_eda.ipynb`: data loading, cleaning checks, missing values,
  duplicates, descriptive statistics, and early risk patterns.
- `notebooks/02_hypothesis_testing.ipynb`: statistical tests for differences
  across provinces, zip codes, gender, and margin segments.
- `notebooks/03_modeling.ipynb`: feature engineering, model training,
  evaluation, and feature-importance interpretation.

Reusable logic should live in `src/` so notebooks stay clean and focused on
analysis, interpretation, and visualization.

## Testing

Tests are stored in `tests/` and can be run locally with:

```bash
pytest
```

GitHub Actions runs the test suite on every push and pull request using
`.github/workflows/ci.yml`.

## Report Structure

The final report should be written in `reports/final_report.md` and include:

- Executive summary.
- Data overview and quality notes.
- Key EDA findings.
- Hypothesis testing results.
- Modeling approach and evaluation metrics.
- Feature-importance insights.
- Business recommendations for ACIS.
- Limitations and next steps.

## Expected Outcome

The final project should provide ACIS with clear, statistically supported
recommendations for identifying low-risk segments, improving pricing decisions,
and understanding the factors most strongly associated with claim risk and
profitability.
