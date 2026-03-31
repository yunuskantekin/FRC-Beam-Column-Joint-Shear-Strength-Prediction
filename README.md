# Machine Learning Based Prediction of Joint Shear Strength for Fiber Reinforced Concrete Beam-Column Connections

**Authors:** Yunus Kantekin, Burcu Burak Bakir, Rodrigo Sarlo

This repository contains the complete source code for the machine learning analysis presented in the paper. Ten regression algorithms are trained and evaluated on a database of fiber reinforced concrete (FRC) beam-column joint specimens, and SHAP-based feature importance analysis is performed to interpret model predictions.

## Models

| Algorithm | Abbreviation |
|---|---|
| Multiple Linear Regression | MLR |
| K-Nearest Neighbors | KNN |
| Support Vector Regression | SVR |
| Decision Tree | DT |
| Random Forest | RF |
| Gradient Boosting | GB |
| XGBoost | XGB |
| LightGBM | LGBM |
| CatBoost | CB |
| Multi-Layer Perceptron | MLP |

## Repository Structure

```
Code/
├── config.py                   # Constants, hyperparameter grids, GPU detection
├── data.py                     # Data loading and train/test split preparation
├── models.py                   # Model training and metric computation
├── evaluation.py               # Results aggregation and representative seed selection
├── shap_analysis.py            # SHAP value computation and figure generation
├── export.py                   # Excel export
├── main.py                     # Entry point
├── ML_FRC_BCJ_Vj_Predictor.py  # GUI prediction tool
├── ML_FRC_BCJ_Vj_Database.xlsx # Dataset
└── requirements.txt            # Python dependencies
```

## Getting Started

### 1. Install dependencies

Python 3.12 or higher is required.

```bash
pip install -r requirements.txt
```

> **Linux users:** tkinter may need to be installed separately:
> ```bash
> sudo apt-get install python3-tk
> ```

### 2. Run the analysis

```bash
cd Code
python main.py
```

### 3. Run the GUI predictor (optional)

A pre-trained `FRC_BCJ_Rep_ML_Models.joblib` is provided in the repository and can be used directly. To generate your own, run `main.py` first. The GUI can then be launched with:

```bash
python ML_FRC_BCJ_Vj_Predictor.py
```

## Outputs

Each run creates a timestamped folder inside `Code/`:

```
Code/
└── Export_TIMESTAMP/
    ├── ML_Detailed_Results_TIMESTAMP.xlsx
    ├── SHAP_Beeswarm_ALG_TIMESTAMP.tiff
    ├── SHAP_Bar_ALG_TIMESTAMP.tiff
    ├── SHAP_Dependence_ALG_TIMESTAMP.tiff
    ├── SHAP_Interaction_ALG_TIMESTAMP.tiff
    ├── SHAP_Compare_ALG_TIMESTAMP.tiff
    └── FRC_BCJ_Rep_ML_Models.joblib
```

| File | Description |
|---|---|
| `ML_Detailed_Results_TIMESTAMP.xlsx` | Performance metrics, per-seed results, predictions, hyperparameters, SHAP summary |
| `SHAP_Beeswarm_ALG_TIMESTAMP.tiff` | SHAP beeswarm plot for the best tree-based model |
| `SHAP_Bar_ALG_TIMESTAMP.tiff` | Mean absolute SHAP value bar chart |
| `SHAP_Dependence_ALG_TIMESTAMP.tiff` | f'c-centered dependence plots |
| `SHAP_Interaction_ALG_TIMESTAMP.tiff` | SHAP interaction matrix |
| `SHAP_Compare_ALG_TIMESTAMP.tiff` | SHAP summary plots for remaining ensemble models |
| `FRC_BCJ_Rep_ML_Models.joblib` | Pretrained models for the GUI predictor |

## Validation Strategy

The analysis uses **100 repeated random holdout** splits (80% training, 20% test) with different random seeds. For each seed, hyperparameters are tuned independently using 5-fold cross-validation on the training set. Final performance metrics are reported as mean ± standard deviation across all 100 seeds.

A **representative seed** is selected as the seed whose models collectively deviate least from the 100-seed mean RMSE across all algorithms simultaneously. SHAP analysis is performed on the representative seed's best tree-based model.

## Notes

- CatBoost uses GPU automatically if CUDA is available; all other models run on CPU.
- SHAP requires the `shap` package. If unavailable, it is skipped and all other outputs are still produced.
- The comparison plots cover all ensemble models except the primary SHAP model.
