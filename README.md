**# DOME: Dynamic Optimization Meta-Ensemble Framework for Inventory-Based Regional Debris-Flow Susceptibility Assessment

## Overview

This repository implements **DOME** (**D**ynamic **O**ptimization **M**eta-**E**nsemble), a computational framework for **inventory-based regional debris-flow susceptibility assessment** in the China-Pakistan Economic Corridor (CPEC).

The framework is designed to address the limitations of single machine-learning models when modeling complex, nonlinear, and heterogeneous debris-flow conditioning relationships across broad regional settings. DOME combines:

- **dynamic learner selection**,
- **stacked meta-ensemble learning**,
- **Greater Cane Rat Algorithm (GCRA)-based optimization**, and
- **SHAP-based interpretation**.

This codebase has been updated to align with the revised manuscript framing. Accordingly, the repository emphasizes **susceptibility assessment** rather than a full formal exposure-vulnerability-risk framework.

> **Terminology note**  
> Some local spreadsheet versions may still use the historical target column name `Risk_index`. In this repository, that legacy name is supported for backward compatibility, but in the revised manuscript context it should be interpreted as the study's **susceptibility-oriented target/proxy**, not as a claim of complete formal risk assessment.

---

## Key Features

- **Inventory-Based Susceptibility Modeling**  
  Designed for regional debris-flow susceptibility assessment using debris-flow inventory data and environmental conditioning factors.

- **Dynamic Learner Selection**  
  Adaptively screens and selects effective base learners and meta-learners based on model performance.

- **Meta-Ensemble Optimization**  
  Uses the **Greater Cane Rat Algorithm (GCRA)** to optimize learner combinations and prior weighting structures.

- **Stacked Ensemble Learning with OOF Meta-Features**  
  Uses out-of-fold predictions to train the meta-learner in a more rigorous way and reduce information leakage.

- **Feature Analysis and Weighting**  
  Integrates **RFE**, **multicollinearity screening** (e.g., VIF/correlation), and **ICWCM-style weighting**.

- **SHAP-Based Interpretation**  
  Supports optional post-hoc interpretation of fitted DOME outputs using SHAP.

- **Command-Line Workflow**  
  Provides training, prediction, and explanation functions through `domecli.py`.

- **Backward-Compatible Target Resolution**  
  Automatically supports common target names such as:
  - `Susceptibility_index`
  - `Susceptibility`
  - `Risk_index`
  - `label`
  - `target`

---

## Repository Structure

```text
DOME-Framework/
├── main.py                            # Core DOME model implementation
├── gcra_optimizer.py                  # Greater Cane Rat Algorithm optimizer
├── utils.py                           # Utility functions (RFE, VIF, ICWCM, metrics, etc.)
├── test.py                            # Model validation and testing script
├── domecli.py                         # Command-line interface
├── requirements.txt                   # Python dependencies
├── CPEC_debris_flow_dataset_3447.xlsx # Working debris-flow dataset
├── LICENSE                            # License file
└── README.md                          # This file
```

---

## Installation

### Requirements

- Python **3.8+** recommended
- `pip` or `conda`

### Clone the Repository

```bash
git clone https://github.com/zjusuge/DOME_Dynamic-Optimization-Meta-Ensemble-Framework-for-Computational-Debris-Flow-Risk-Assessment.git
cd DOME_Dynamic-Optimization-Meta-Ensemble-Framework-for-Computational-Debris-Flow-Risk-Assessment
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Minimal Manual Installation

```bash
pip install numpy pandas scipy scikit-learn statsmodels openpyxl joblib
```

### Optional Packages

Install these if you want enhanced learner diversity and model interpretation:

```bash
pip install shap xgboost lightgbm
```

---

## Quick Start

### 1. Run the validation script

```bash
python test.py
```

### 2. Train the model from the command line

```bash
python domecli.py train \
  --data CPEC_debris_flow_dataset_3447.xlsx \
  --target Risk_index \
  --out results
```

### 3. Generate predictions

```bash
python domecli.py predict \
  --model-path results/model.joblib \
  --input CPEC_debris_flow_dataset_3447.xlsx \
  --target Risk_index \
  --out predictions.csv
```

### 4. Generate SHAP explanations

```bash
python domecli.py explain \
  --model-path results/model.joblib \
  --input CPEC_debris_flow_dataset_3447.xlsx \
  --target Risk_index \
  --out explain_results
```

---

## Python Usage Example

```python
import pandas as pd
from main import DOMEModel, resolve_target_column

# Load dataset
df = pd.read_excel("CPEC_debris_flow_dataset_3447.xlsx")

# Automatically resolve target column
target_col = resolve_target_column(df, preferred="Risk_index")

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Initialize DOME
dome = DOMEModel(
    alpha=0.34,
    beta=0.04,
    gamma=0.01,
    random_state=42,
    cv_splits=5
)

# Train model
results = dome.fit(X, y)

# Predict susceptibility values
predictions = dome.predict(X)

# View metrics
metrics = results["performance_metrics"]
print(f"RMSE: {metrics['RMSE']:.6f}")
print(f"MAE: {metrics['MAE']:.6f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"Spearman Correlation: {metrics['Spearman_Correlation']:.6f}")
```

### Optional SHAP Interpretation

```python
try:
    shap_result = dome.get_shap_explanations(
        X.head(50),
        background_size=20,
        explain_size=20,
        nsamples=100
    )
    print(shap_result["summary"].head())
except ImportError:
    print("Install 'shap' to enable SHAP-based interpretation.")
```

---

## Methodological Workflow

The DOME framework follows a susceptibility-oriented computational workflow:

1. **Data preprocessing**  
   Clean, coerce, and standardize the conditioning factors and target values.

2. **Training/testing partitioning**  
   Split the dataset into training and testing subsets.

3. **Feature analysis and selection**  
   Apply recursive feature elimination (RFE), multicollinearity screening, and feature weighting.

4. **Initial learner screening**  
   Evaluate candidate base learners using cross-validation and construct out-of-fold meta-features.

5. **Dynamic optimization**  
   Use GCRA to optimize the learner combination and prior weighting structure.

6. **Stacked model training**  
   Train the selected base learners and meta-learner using weighted stacking.

7. **Model validation**  
   Evaluate the fitted model using:
   - RMSE
   - MAE
   - MAPE
   - Spearman correlation

8. **Model interpretation**  
   Summarize selected learners, feature importance, and optional SHAP-based interpretation.

---

## Dataset

This repository is designed around the working spreadsheet:

- `CPEC_debris_flow_dataset_3447.xlsx`

The current manuscript treats the dataset as an **inventory-based regional debris-flow susceptibility assessment dataset** for the China-Pakistan Economic Corridor.

### Conditioning Factors Used in the Manuscript / Workbook

#### Topographical Factors

- **X1** — Basin area (km²)
- **X2** — Average elevation (m)
- **X3** — Relative height difference (m)
- **X4** — Maximum slope (°)
- **X5** — Average slope (°)
- **X6** — Main channel bed gradient (m/km)

#### Geological Factors

- **X7** — Lithology (-)
- **X8** — Fault density (-)

#### Climatic Factors

- **X9** — Annual precipitation (mm)
- **X10** — Maximum temperature (°C)

#### Hydrological Factors

- **X11** — River network density (-)

#### Socioeconomic / Infrastructure-Related Factors

- **X12** — Electricity consumption (kWh)
- **X13** — GDP (USD/km²)
- **X14** — Population density (people/km²)
- **X15** — Road network density (km/km²)

> These factor labels are retained from the manuscript/workbook schema currently used in this repository.

---

## Data Source

The baseline debris-flow inventory reference is derived from the China Scientific Data repository:

**Jiang, W., Lin, G., Wang, T., et al. (2021).**  
*A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor*.  
**China Scientific Data**.  
DOI: <https://doi.org/10.11922/11-6035.csd.2021.0069.zh>

The source page title provided in the SciEngine record is:

> **A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor**

---

## Command-Line Interface

The repository includes a command-line interface through `domecli.py`.

### Train

```bash
python domecli.py train \
  --data CPEC_debris_flow_dataset_3447.xlsx \
  --target Risk_index \
  --cv 5 \
  --seed 42 \
  --out results
```

### Predict

```bash
python domecli.py predict \
  --model-path results/model.joblib \
  --input new_data.xlsx \
  --out predictions.csv
```

### Explain

```bash
python domecli.py explain \
  --model-path results/model.joblib \
  --input CPEC_debris_flow_dataset_3447.xlsx \
  --target Risk_index \
  --out explain_results
```

---

## Output Files

Typical outputs may include:

- `training_results.json`
- `model_summary.json`
- `metrics.json`
- `feature_importance.csv`
- `model.joblib`
- `predictions.csv`
- `shap_summary.csv`
- `shap_metadata.json`
- `test_results.json`
- `extended_test_results.json`

---

## Reproducibility

To improve reproducibility:

1. Use a clean Python environment (`venv` or `conda`)
2. Install dependencies from `requirements.txt`
3. Set the random seed, e.g. `random_state=42`
4. Keep the dataset schema unchanged
5. Use the same target-column definition across experiments
6. Record any optional dependency versions, especially:
   - `xgboost`
   - `lightgbm`
   - `shap`

---

## Notes on Interpretation Scope

This repository focuses on **susceptibility modeling and interpretation**.

It does **not**, by itself, constitute:

- a complete hazard-frequency model,
- a full exposure-vulnerability-risk framework,
- or a direct policy decision engine.

If needed, downstream integration with exposure, vulnerability, or infrastructure prioritization can be conducted **after** susceptibility modeling.

---

## Citation

If you use this repository in academic work, please cite both the dataset source and the associated manuscript/repository.

### Dataset Citation

**Jiang, W., Lin, G., Wang, T., et al. (2021).**  
*A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor*.  
**China Scientific Data**.  
DOI: <https://doi.org/10.11922/11-6035.csd.2021.0069.zh>

### Repository / Framework Citation

If you are preparing a manuscript based on this code, you may cite the corresponding DOME study once the bibliographic details are finalized.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Author

**Tianlong Wang**

- Ocean College, Zhejiang University, Zhoushan 316000, China
- School of Civil and Environmental Engineering, Nanyang Technological University, Singapore 637616, Singapore
- Contact: <tianlong_wang@zju.edu.cn>**
