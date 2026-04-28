# xRFM vs XGBoost vs Random Forest — Empirical Study

## Overview

This project compares three machine learning models:

* **xRFM** (experimental model)
* **XGBoost**
* **Random Forest**

across multiple datasets, focusing on:

* Predictive performance
* Training and inference time
* Interpretability (AGOP vs standard methods)
* Scalability

---

## Datasets

| Dataset  | Task           |
| -------- | -------------- |
| Diabetes | Classification |
| Housing  | Regression     |
| Wine     | Regression     |
| Steel    | Classification |
| Shoppers | Classification |

---

## Features

### 1. Model Benchmarking

* Hyperparameter tuning via random search
* Train/validation/test split
* Metrics:

  * Accuracy (classification)
  * RMSE (regression)
  * Runtime

---

### 2. Interpretability Analysis

Compares feature importance methods:

* **AGOP (xRFM-specific)**
* PCA importance
* Mutual Information
* Permutation importance

Outputs:

* Feature importance plots
* Rank correlation between methods

---

### 3. Scaling Experiment

Tests how models behave as dataset size increases.

Outputs:

* Performance vs dataset size
* Runtime vs dataset size

---

## Project Structure

```
src/
│
├── datasets.py        # Data loading & preprocessing
├── models.py          # Model training + tuning
├── evaluate.py        # Metrics
├── interpretability.py# Feature importance methods
├── scaling.py         # Scaling experiments
│
main.py                # Main experiment pipeline
results/               # Output CSVs and plots
data/                  # Raw datasets
README.md              # Introductory file
requirements.txt       # Required packages
```

---

## Key Design Decisions

### xRFM Subsampling

xRFM has **O(n²)** memory complexity.

To prevent crashes:

* Training is limited to 5,000 samples
* Random subsampling is applied

---

### Preprocessing Pipeline

* Numerical features → StandardScaler
* Categorical features → OneHotEncoder
* High-cardinality features are dropped

---

## How to Run

```bash
python main.py
```

---

## Outputs

* `results/main_table.csv` → performance summary
* `results/*_interpretability.png` → feature importance plots
* Scaling plots

---

## Key Insights (Expected)

* XGBoost typically best predictive performance
* Random Forest stable baseline
* xRFM offers **unique interpretability via AGOP**
* Scaling highlights computational trade-offs

---

## Future Improvements

* Replace random search with Bayesian optimisation
* Add SHAP comparisons
* Parallelise training
* Improve xRFM scalability

---

## Authors
Yatha Shah (z5480800)
Liam Lightfoot (z5589045)
Archana Adlakha (z5694760)
Zac Rose (z5309428)
