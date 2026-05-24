# Insurance Risk Analytics

This project is an end-to-end insurance analytics workflow for AlphaCare Insurance Solutions (ACIS). The goal is to turn raw car-insurance data into practical, evidence-backed business decisions for risk-based pricing and targeted marketing.

ACIS wants to identify low-risk customer, geographic, and vehicle segments so it can improve pricing strategy, reduce claim exposure, and support more profitable growth.

## Project Objectives

The assignment combines exploratory analysis, reproducible data workflows, statistical testing, predictive modeling, and business reporting.

The main objectives are to:

- Understand the structure, quality, and key patterns in the insurance data.
- Identify risk differences across provinces, zip codes, gender groups, vehicles, and insurance plan features.
- Use statistical hypothesis testing to validate whether observed risk differences are meaningful.
- Build predictive models for claim severity and risk-based pricing decisions.
- Interpret model outputs to identify the most important risk drivers.
- Produce a polished final report with clear recommendations for ACIS.

## Business Questions

This project aims to answer questions such as:

- Which provinces have higher or lower claim risk?
- Do men and women differ significantly in insurance risk?
- Do some zip codes produce better margins than others?
- Which vehicles or plan features are linked to high claims?
- Can claim severity be predicted well enough to support pricing decisions?
- Which features matter most in the predictive models?

## Project Workflow

### 1. Exploratory Data Analysis

The EDA phase focuses on understanding the raw insurance data, detecting data quality issues, and discovering early risk patterns.

Key tasks include:

- Data loading and inspection.
- Missing value and outlier analysis.
- Univariate and bivariate analysis.
- Loss ratio and claim severity exploration.
- Geographic, demographic, vehicle, and plan-based risk analysis.

### 2. Data Version Control

DVC is used to make the data workflow reproducible and auditable. Since raw data should not be tracked directly by Git, the `data/` directory is intended to be managed with DVC.

Key tasks include:

- Track datasets with DVC.
- Define reproducible pipeline stages in `dvc.yaml`.
- Keep Git history clean while preserving data lineage.
- Support repeatable analysis and modeling runs.

### 3. Hypothesis Testing

The hypothesis testing phase validates whether risk differences across customer and geographic segments are statistically meaningful.

Example tests include:

- Risk differences across provinces.
- Risk differences across zip codes.
- Risk differences by gender.
- Margin differences across selected segments.

### 4. Predictive Modeling

The modeling phase builds machine learning models to support claim-severity prediction and risk-based premium decisions.

Key tasks include:

- Feature engineering.
- Train-test splitting.
- Model training and evaluation.
- Claim severity prediction.
- Pricing or premium-risk modeling.
- Feature importance interpretation.

## Repository Structure

```text
insurance-risk-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                     # Tracked by DVC, not Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── eda_utils.py
│   ├── hypothesis_tests.py
│   └── modeling.py
├── reports/
│   └── final_report.md
├── tests/
├── .dvc/
├── .gitignore
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Expected Final Deliverables

The final project should include:

- Clean, modular Python code in `src/`.
- Well-organized notebooks for EDA, hypothesis testing, and modeling.
- DVC-tracked data and reproducible pipeline definitions.
- Statistical evidence for key business hypotheses.
- Predictive models with clear evaluation results.
- Model interpretation showing the most important risk drivers.
- A polished final report in `reports/final_report.md`.
- A clean Git history showing steady project progress.

## Business Outcome

The final output should help ACIS make better decisions about pricing, marketing, and risk selection by identifying low-risk segments and explaining the factors most strongly associated with claim risk and profitability.
