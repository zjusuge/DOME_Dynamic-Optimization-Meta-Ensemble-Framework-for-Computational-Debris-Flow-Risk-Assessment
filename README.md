# DOME: Dynamic Optimization Meta-Ensemble Framework for Computational Debris Flow Risk Assessment

## Overview
This project implements the DOME (Dynamic Optimization Meta-Ensemble) framework for computational debris flow risk prediction. DOME addresses the limitations of traditional machine learning models in capturing nonlinear dynamics and generalizing across diverse geological and climatic conditions through dynamic learner selection and multi-objective optimization using the Greater Cane Rat Algorithm (GCRA).

## Key Features
- **Dynamic Learner Selection**: Adaptively adjusts base and meta-learner combinations based on data characteristics
- **Multi-objective Optimization**: Balances prediction accuracy, model complexity, learner diversity, and computational efficiency
- **Meta-Ensemble Learning**: Weighted stacking integration of multiple base learners through optimized aggregation
- **Robust Performance**: Maintains stability across varying environmental conditions and data perturbations

## Project Structure
```
DOME-Framework/
├── main.py                            # Main DOME model implementation with 8-step workflow
├── gcra_optimizer.py                  # Greater Cane Rat Algorithm for dynamic optimization
├── utils.py                           # Utility functions (RFE, VIF, ICWCM, evaluation metrics)
├── test.py                            # Model testing and performance evaluation
├── requirements.txt                   # Python dependencies
├── CPEC_debris_flow_dataset_3447.xlsx # Complete debris flow dataset
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## Installation
**Requirements:** Python 3.7+ (3.8+ recommended)

### Clone and Install
```bash
git clone https://github.com/zjusuge/DOME_Dynamic-Optimization-Meta-Ensemble-Framework-for-Computational-Debris-Flow-Risk-Assessment.git
cd DOME_Dynamic-Optimization-Meta-Ensemble-Framework-for-Computational-Debris-Flow-Risk-Assessment
pip install -r requirements.txt
```
### Alternative: Manual Installation
```bash
pip install numpy pandas scipy scikit-learn statsmodels openpyxl xgboost lightgbm
```

## Quick Start
```bash
python test.py
```

## Usage
```python
from main import DOMEModel
import pandas as pd

# Load data
df = pd.read_excel('CPEC_debris_flow_dataset_3447.xlsx')
X = df.drop('Risk_index', axis=1)
y = df['Risk_index']

# Initialize and train model
dome = DOMEModel(alpha=0.34, beta=0.04, gamma=0.01, random_state=42)
results = dome.fit(X, y)

# Make predictions
predictions = dome.predict(X)

# View metrics
metrics = results['performance_metrics']
print(f"RMSE: {metrics['RMSE']:.6f}")
print(f"MAE: {metrics['MAE']:.6f}")
print(f"MAPE: {metrics['MAPE']:.2f}%")
print(f"Spearman Correlation: {metrics['Spearman_Correlation']:.6f}")
```

## Dataset
The framework utilizes **3,447 debris flow records** from the China-Pakistan Economic Corridor (CPEC) region spanning 1961-2016.

### Risk Assessment Factors (15 Features)

**Topographical Factors**
- **X1** - Basin area (km²)
- **X2** - Average elevation (m)
- **X3** - Relative height difference (m)
- **X4** - Maximum slope (°)
- **X5** - Average slope (°)
- **X6** - Main channel bed gradient (m/km)

**Geological Factors**
- **X7** - Lithology (-)
- **X8** - Fault density (-)

**Climatic Factors**
- **X9** - Annual precipitation (mm)
- **X10** - Maximum temperature (°C)

**Hydrological Factors**
- **X11** - River network density (-)

**Socio-economic Factors**
- **X12** - Electricity consumption (kWh)
- **X13** - GDP (USD/km²)
- **X14** - Population density (people/km²)
- **X15** - Road network density (km/km²)

## Methodology
DOME implements an 8-step computational workflow:

1. **Data Preprocessing**: Data collection, cleaning, and normalization
2. **Feature Selection**: Recursive Feature Elimination (RFE) combined with multicollinearity detection (VIF/correlation)
3. **Feature Weighting**: Information-Correlation Weighted Combination Method (ICWCM) for optimal feature weighting
4. **Initial Learner Selection**: Evaluation of base and meta learners via cross-validation
5. **Dynamic Optimization**: GCRA optimization for dynamic learner selection and weighting
6. **Model Training**: Weighted stacking with optimized learner combinations
7. **Model Validation**: Performance evaluation using RMSE, MAE, MAPE, and Spearman correlation
8. **Result Analysis**: Interpretation of optimization process and model results

## Dependencies
Core dependencies listed in `requirements.txt`:
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
openpyxl>=3.0.0
xgboost>=1.5.0 # Optional
lightgbm>=3.3.0 # Optional
```

## Reproducibility
To ensure reproducible results:
1. Use a clean virtual environment (e.g., `venv` or `conda`)
2. Install exact package versions from `requirements.txt`
3. Set `random_state=42` (default) in `DOMEModel` initialization
4. Use the provided dataset for benchmarking

## Data Source
The baseline debris flow inventory data is derived from the China Scientific Data repository:
**Citation**: Jiang, W., Lin, G., Wang, T., et al. (2021). A dataset of distributions and characteristics of debris flows in the China-Pakistan Economic Corridor. *China Scientific Data*, 6(4). https://doi.org/10.11922/11-6035.csd.2021.0069.zh

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Tianlong Wang
- Ocean College, Zhejiang University, Zhoushan 316000, China
- School of Civil and Environmental Engineering, Nanyang Technological University, Singapore 637616, Singapore
- Contact: tianlong_wang@zju.edu.cn
