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
│   ├── interim_report.md
│   ├── final_report.md
│   ├── hypothesis_test_results.csv
│   ├── model_comparison.csv
│   ├── claim_frequency_model_metrics.csv
│   ├── pricing_framework_sample.csv
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_preparation.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   ├── modeling.py
│   └── model_interpretability.py
├── tests/
│   ├── __init__.py
│   ├── test_eda_utils.py
│   ├── test_hypothesis_tests.py
│   ├── test_modeling.py
│   ├── test_model_interpretability.py
│   └── test_project_setup.py
├── .gitignore
├── dvc.yaml
├── pyproject.toml
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

Raw and processed datasets are stored under `data/` and tracked with DVC instead
of Git. This keeps the Git repository lightweight while preserving data lineage
and reproducibility.

Current data pipeline:

```text
data/raw/insurance_data.csv -> data/processed/insurance_data_cleaned.csv
```

The `prepare_data` stage is defined in `dvc.yaml` and runs:

```bash
python -m src.data_preparation --input data/raw/insurance_data.csv --output data/processed/insurance_data_cleaned.csv
```

Useful DVC commands:

```bash
dvc status
dvc repro
dvc push
dvc pull
```

The current local remote is named `localstorage` and points to
`..\acis-dvc-storage`.

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

## Task 1 Methodology: Exploratory Data Analysis

Task 1 focuses on understanding the ACIS insurance dataset, checking data
quality, and identifying early risk and profitability patterns before formal
hypothesis testing and modeling.

### Dataset Loading

The dataset is loaded in `notebooks/01_eda.ipynb` using
`src.data_loader.load_insurance_data`. This helper safely reads common file
formats such as CSV, TXT, Excel, Parquet, and JSON, then validates that the file
exists, the format is supported, and the loaded dataset is not empty.

By default, the notebook expects:

```bash
data/raw/insurance_data.csv
```

If the dataset has a different filename or delimiter, update the `DATA_PATH`
variable or pass the correct pandas reader options, such as `sep="|"`.

### EDA Steps Performed

The EDA notebook performs the following steps:

- Load and preview the insurance dataset.
- Summarize dataset shape, column types, missing cells, and duplicate rows.
- Inspect numeric and categorical columns.
- Review descriptive statistics for `TotalPremium`, `TotalClaims`,
  `CustomValueEstimate`, and other numeric fields.
- Calculate overall loss ratio and underwriting margin.
- Add row-level risk metrics for margin and loss ratio.
- Visualize distributions with histograms.
- Detect potential outliers with box plots.
- Explore numeric relationships with a correlation heatmap.
- Compare risk and profitability by province, vehicle type, gender, car make,
  and car model.

### Loss Ratio and Margin

Loss ratio measures how much of the collected premium is consumed by claims:

```text
loss ratio = total claims / total premium
```

A lower loss ratio usually indicates a more profitable or lower-risk segment.
A high loss ratio may indicate underpricing, high claim frequency, high claim
severity, or a segment that needs underwriting review.

Margin measures the difference between premium collected and claims paid:

```text
margin = total premium - total claims
```

Positive margin suggests that premiums exceed claims. Negative margin suggests
that claims are greater than premiums and that the segment may require pricing
or risk-management action.

### Missing Values and Outliers

Missing values are summarized by count and percentage for every column. The EDA
does not automatically drop missing records because missingness may carry
business meaning or affect important fields such as premium, claims, location,
vehicle attributes, or customer demographics.

Outliers are identified using descriptive statistics, histograms, and box plots.
They are not removed automatically because large claims or high-value vehicles
may represent real insurance risk. Any outlier treatment should be justified and
documented before modeling.

### Running the Notebook

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/01_eda.ipynb
```

Run the cells from top to bottom. If the dataset is not named
`insurance_data.csv`, update `DATA_PATH` in the notebook before running the data
loading cell.

### Expected EDA Insights

The EDA is expected to produce early evidence about:

- Overall data quality and fields that need cleaning.
- Premium, claim, and vehicle-value distributions.
- Whether claims are concentrated among a small number of policies.
- Overall loss ratio and margin performance.
- Provinces with relatively high or low risk.
- Vehicle types, makes, and models associated with higher claims.
- Gender-level patterns that should be validated through hypothesis testing.
- Outliers and skewed variables that may affect predictive modeling.

## Testing

Tests are stored in `tests/` and can be run locally with:

```bash
ruff check .
pytest
```

GitHub Actions runs Ruff linting and the test suite on every push and pull
request using `.github/workflows/ci.yml`.

## Hypothesis Testing

`notebooks/02_hypothesis_testing.ipynb` tests the required ACIS hypotheses using
the reusable helpers in `src/hypothesis_tests.py`.

The notebook covers:

- Claim-frequency differences across provinces.
- Claim-frequency differences between high-volume provinces.
- Margin differences between high-volume zip codes.
- Claim-frequency and margin differences by gender.

Results are exported to `reports/hypothesis_test_results.csv`.

## Modeling and Pricing

`notebooks/03_modeling.ipynb` supports both claim severity and claim frequency:

- Severity models estimate `TotalClaims` for policies with claims.
- Frequency models estimate `P(claim)` for all policies.
- Risk-based pricing combines both outputs:

```text
Pure Premium = P(claim) x Predicted Severity
Technical Premium = Pure Premium + Expense Loading + Risk Load + Profit Margin
```

Modeling outputs are saved in `reports/`, including model comparison,
claim-frequency metrics, pricing samples, and interpretability summaries.

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

## Git Workflow and Submission

The project uses task branches:

- `task-1`: EDA scaffold and exploratory analysis
- `task-2`: DVC pipeline and data preparation
- `task-3`: Hypothesis testing
- `task-4`: Modeling, interpretability, pricing, and final reports

The complete end-to-end deliverable lives on **`task-4`**.

### What to push

Push the **`task-4`** branch to GitHub:

```bash
git checkout task-4
git push -u origin task-4
```

Then open a pull request from **`task-4` → `main`** on GitHub and merge it when
CI passes. This makes the full project visible on the default branch and counts
toward your GitHub contribution graph.

Do **not** commit raw or processed CSV files. They are tracked with DVC:

```bash
dvc push
```

Use this only if your reviewer needs access to the DVC remote storage.
