# DOME: Dynamic Optimization Meta-Ensemble Framework for Computational Debris Flow Risk Assessment

## OVERVIEW

This project implements the DOME (Dynamic Optimization Meta-Ensemble) framework for computational debris flow risk prediction. DOME addresses the limitations of traditional machine learning models in capturing nonlinear dynamics and generalizing across diverse geological and climatic conditions through dynamic learner selection and multi-objective optimization using the Greater Cane Rat Algorithm (GCRA).

## KEY FEATURES

- Dynamic Learner Selection: Adaptively adjusts base and meta-learner combinations based on data characteristics
- Multi-objective Optimization: Balances prediction accuracy, model complexity, learner diversity, and computational efficiency
- Meta-Ensemble Learning: Weighted stacking integration of multiple base learners through optimized aggregation
- Robust Performance: Maintains stability across varying environmental conditions and data perturbations

## PROJECT STRUCTURE

- `main.py` — Main DOME model implementation with 8-step workflow  
- `gcra_optimizer.py` — Greater Cane Rat Algorithm for dynamic optimization  
- `utils.py` — Utility functions (RFE, VIF, ICWCM, evaluation metrics)  
- `test.py` — Model testing and performance evaluation  
- `CPEC_debris_flow_dataset_3447.xlsx` — Complete debris flow dataset  
- `README.md` — This file

## INSTALLATION

Recommended (Python 3.8+):

```bash
# Option 1: install from requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: quick install (same effect; XGBoost/LightGBM optional)
pip install numpy pandas scipy scikit-learn statsmodels openpyxl et_xmlfile xgboost lightgbm
If you must use Python 3.7, please pin versions accordingly (ask for a 3.7-compatible spec).

DATASET
The framework utilizes 3,447 debris flow records from the China-Pakistan Economic Corridor (CPEC) region spanning 1961–2016. The dataset integrates multi-source data for comprehensive risk assessment.

Risk Assessment Factors (15 features)
Topographical Factors

X1 — Basin area (km²)
X2 — Average elevation (m)
X3 — Relative height difference (m)
X4 — Maximum slope (°)
X5 — Average slope (°)
X6 — Main channel bed gradient (m/km)
Geological Factors

X7 — Lithology (-)
X8 — Fault density (-)
Climatic Factors

X9 — Annual precipitation (mm)
X10 — Maximum temperature (°C)
Hydrological Factors

X11 — River network density (-)
Socio-economic Factors

X12 — Electricity consumption (kWh)
X13 — GDP (USD/km²)
X14 — Population density (people/km²)
X15 — Road network density (km/km²)
USAGE
Quick Start:

bash

复制
python test.py
Custom Usage:

python

运行

复制
from main import DOMEModel
import pandas as pd

# Load your data
df = pd.read_excel('your_data.xlsx')
X = df.drop('Risk_index', axis=1)
y = df['Risk_index']

# Initialize and train DOME model
dome = DOMEModel()
results = dome.fit(X, y)

# Make predictions
predictions = dome.predict(X)  # or X_new with same columns

# Access metrics
metrics = results['performance_metrics']
print(metrics)
METHODOLOGY
DOME implements an 8-step computational workflow:

Data preprocessing and normalization
Recursive Feature Elimination (RFE) + multicollinearity detection (VIF/correlation)
Information–Correlation Weighted Combination Method (ICWCM) for feature weighting
Base/meta learner evaluation via cross-validation
GCRA optimization for dynamic learner selection and weighting
Model training and optimization (weighted stacking)
Model performance evaluation (RMSE/MAE/MAPE/Spearman)
Optimization process analysis and interpretation
TECHNICAL HIGHLIGHTS
Greater Cane Rat Algorithm (GCRA): Bio-inspired metaheuristic for global optimization
Dynamic ensemble learning: Adaptive learner combination based on data characteristics
Multi-objective design: Considers prediction error, model complexity, and computational cost
Weighted stacking: Optimized aggregation of base learner predictions
Robust validation: K-fold cross-validation
REQUIREMENTS
Python 3.8+ recommended
Exact package specifications (same as requirements.txt):

txt

复制
numpy>=1.22,<2.0
pandas>=1.4
scipy>=1.8
scikit-learn>=1.1
statsmodels>=0.13
openpyxl>=3.0
et_xmlfile>=1.1
# Optional (enabled automatically if installed)
xgboost>=1.6
lightgbm>=3.3
Notes:

The code uses statsmodels (VIF) and et_xmlfile (Excel I/O dependency).
XGBoost and LightGBM are optional; the code will include them if installed (try-import).
REPRODUCIBILITY
Use a clean virtual environment (e.g., venv or conda).
Install exact requirements as above for consistent results.
Set random_state in DOMEModel for reproducibility (already defaulted to 42).
CITATION AND DATA SOURCE
Jiang, W., Lin, G., Wang, T., et al. (2021). A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor. China Scientific Data, 6(4). https://doi.org/10.11922/11-6035.csd.2021.0069.zh
The original dataset provides foundational debris flow characteristics for the CPEC region. Additional socio-economic factors (X12–X15) were integrated from multiple sources to enhance comprehensiveness.

LICENSE
Unless otherwise specified, this repository is for academic and research use. Please verify dataset licensing from the original source before redistribution.

AUTHOR
Tianlong Wang
Ocean College, Zhejiang University, Zhoushan 316000, China
School of Civil and Environmental Engineering, Nanyang Technological University, Singapore 637616, Singapore

Contact: tianlong_wang@zju.edu.cn


复制
17:08
你给我的有的是markdown有是text有的是py格式

17:09

GPT-5

markdown

复制
# DOME: Dynamic Optimization Meta-Ensemble Framework for Computational Debris Flow Risk Assessment

## OVERVIEW
This project implements the DOME (Dynamic Optimization Meta-Ensemble) framework for computational debris flow risk prediction. DOME addresses the limitations of traditional machine learning models in capturing nonlinear dynamics and generalizing across diverse geological and climatic conditions through dynamic learner selection and multi-objective optimization using the Greater Cane Rat Algorithm (GCRA).

## KEY FEATURES
- Dynamic Learner Selection: Adaptively adjusts base and meta-learner combinations based on data characteristics
- Multi-objective Optimization: Balances prediction accuracy, model complexity, learner diversity, and computational efficiency
- Meta-Ensemble Learning: Weighted stacking integration of multiple base learners through optimized aggregation
- Robust Performance: Maintains stability across varying environmental conditions and data perturbations

## PROJECT STRUCTURE
- `main.py` — Main DOME model implementation with 8-step workflow  
- `gcra_optimizer.py` — Greater Cane Rat Algorithm for dynamic optimization  
- `utils.py` — Utility functions (RFE, VIF, ICWCM, evaluation metrics)  
- `test.py` — Model testing and performance evaluation  
- `CPEC_debris_flow_dataset_3447.xlsx` — Complete debris flow dataset  
- `README.md` — This file

## INSTALLATION
Recommended (Python 3.8+):

```bash
# Option 1: install from requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: quick install (same effect; XGBoost/LightGBM optional)
pip install numpy pandas scipy scikit-learn statsmodels openpyxl et_xmlfile xgboost lightgbm
If you must use Python 3.7, pin compatible versions (ask for a 3.7-specific spec).

DATASET
The framework utilizes 3,447 debris flow records from the China-Pakistan Economic Corridor (CPEC) region spanning 1961–2016. The dataset integrates multi-source data for comprehensive risk assessment.

Risk Assessment Factors (15 features)
Topographical Factors

X1 — Basin area (km²)
X2 — Average elevation (m)
X3 — Relative height difference (m)
X4 — Maximum slope (°)
X5 — Average slope (°)
X6 — Main channel bed gradient (m/km)
Geological Factors

X7 — Lithology (-)
X8 — Fault density (-)
Climatic Factors

X9 — Annual precipitation (mm)
X10 — Maximum temperature (°C)
Hydrological Factors

X11 — River network density (-)
Socio-economic Factors

X12 — Electricity consumption (kWh)
X13 — GDP (USD/km²)
X14 — Population density (people/km²)
X15 — Road network density (km/km²)
USAGE
Quick Start:

bash

复制
python test.py
Custom Usage:

python

运行

复制
from main import DOMEModel
import pandas as pd

# Load your data
df = pd.read_excel('your_data.xlsx')
X = df.drop('Risk_index', axis=1)
y = df['Risk_index']

# Initialize and train DOME model
dome = DOMEModel()
results = dome.fit(X, y)

# Make predictions
predictions = dome.predict(X)  # or X_new with same columns

# Access metrics
metrics = results['performance_metrics']
print(metrics)
METHODOLOGY
DOME implements an 8-step computational workflow:

Data preprocessing and normalization
Recursive Feature Elimination (RFE) + multicollinearity detection (VIF/correlation)
Information–Correlation Weighted Combination Method (ICWCM) for feature weighting
Base/meta learner evaluation via cross-validation
GCRA optimization for dynamic learner selection and weighting
Model training and optimization (weighted stacking)
Model performance evaluation (RMSE/MAE/MAPE/Spearman)
Optimization process analysis and interpretation
TECHNICAL HIGHLIGHTS
Greater Cane Rat Algorithm (GCRA): Bio-inspired metaheuristic for global optimization
Dynamic ensemble learning: Adaptive learner combination based on data characteristics
Multi-objective design: Considers prediction error, model complexity, and computational cost
Weighted stacking: Optimized aggregation of base learner predictions
Robust validation: K-fold cross-validation
REQUIREMENTS
Python 3.8+ recommended
Exact package specifications (store the same content in requirements.txt):

text

复制
numpy>=1.22,<2.0
pandas>=1.4
scipy>=1.8
scikit-learn>=1.1
statsmodels>=0.13
openpyxl>=3.0
et_xmlfile>=1.1
# Optional (enabled automatically if installed)
xgboost>=1.6
lightgbm>=3.3
Notes:

The code uses statsmodels (VIF) and et_xmlfile (Excel I/O dependency).
XGBoost and LightGBM are optional; the code will include them if installed (try-import).
REPRODUCIBILITY
Use a clean virtual environment (e.g., venv or conda).
Install exact requirements as above for consistent results.
Random seeds are set via random_state in DOMEModel (default 42).
DATA SOURCE
Jiang, W., Lin, G., Wang, T., et al. (2021). A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor. China Scientific Data, 6(4). https://doi.org/10.11922/11-6035.csd.2021.0069.zh

The original dataset provides foundational debris flow characteristics for the CPEC region. Additional socio-economic factors (X12–X15) were integrated from multiple sources to enhance comprehensiveness.

LICENSE
Unless otherwise specified, this repository is for academic and research use. Please verify dataset licensing from the original source before redistribution.

AUTHOR
Tianlong Wang
Ocean College, Zhejiang University, Zhoushan 316000, China
School of Civil and Environmental Engineering, Nanyang Technological University, Singapore 637616, Singapore

Contact: tianlong_wang@zju.edu.cn