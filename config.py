import subprocess

def _detect_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("GPU detected.")
            return True
    except Exception:
        pass
    print("No GPU detected.")
    return False

USE_GPU = _detect_gpu()

RANDOM_SEEDS = list(range(100)) # 100 repeated random splits (seeds 0-99)
TEST_SIZE = 0.20 # 80/20 train/test split
DATA_FILE = 'ML_FRC_BCJ_Vj_Database.xlsx' # Dataset excel file

TARGET = 'vj_test' # Output: joint shear strength

SELECTED_FEATURES = ['jtype', 'fbrtype', 'hb', 'bb', 'hc', 'bc',
    'fc', 'fyv', 'rhob', 'rhoc', 'rhov', 'vf', 'ar', 'ftf', 'n'] # Inputs

CATEGORICAL_FEATURES = ['jtype', 'fbrtype'] # Categorical inputs

# Parameters stand for:

# jtype: Joint type (Interior/Exterior)
# fbrtype: Fiber type (HESF, S, PVA, Hybrid, PE, PP, PET, Aramid, Basalt)
# hb: Beam depth (mm)
# bb: Beam width (mm)
# hc: Column depth (mm)
# bc: Column width (mm)
# fc: Composite compressive strength (MPa)
# fyv: Joint transverse reinforcement yield strength (MPa)
# rhob: Beam longitudinal reinforcement ratio
# rhoc: Column longitudinal reinforcement ratio
# rhov: Joint transverse reinforcement ratio
# vf: Fiber volume fraction
# ar: Fiber aspect ratio (L/D)
# ftf: Fiber tensile strength (MPa)
# n: Column axial load ratio

SHAP_LABELS = {'fc': "f'c", 'jtype': 'Joint Type', 'fbrtype': 'Fiber Type',
    'hb': 'hb', 'bb': 'bb', 'hc': 'hc', 'bc': 'bc', 'fyv': 'fyv',
    'rhob': 'ρb', 'rhoc': 'ρc', 'rhov': 'ρv', 'vf': 'Vf', 'ar': 'Lf/Df',
    'ftf': 'ftf', 'n': 'n',}

_SHAP_TREE_MODELS = ['LGBM', 'XGB', 'RF', 'GB', 'DT']

PARAM_GRIDS = {
    "MLR": {},
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVR": {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1],
        "epsilon": [0.05, 0.1, 0.2],
    },
    "DT": {
        "criterion": ["squared_error"],
        "max_depth": [8, 10, 12],
        "min_samples_split": [8, 12, 16],
        "min_samples_leaf": [4, 6, 8],
        "max_features": ["sqrt"],
    },
    "RF": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "GB": {
        "loss": ["squared_error", "absolute_error"],
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "subsample": [0.7, 1.0],
        "max_features": [None, "sqrt"],
    },
    "XGB": {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1, 10],
    },
    "LGBM": {
        "n_estimators": [300, 500, 700, 900],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [20, 30, 50],
        "max_depth": [-1, 6, 10],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "min_child_samples": [10, 20, 40],
    },
    "CB": {
        "iterations": [200, 400],
        "learning_rate": [0.03, 0.05],
        "depth": [4, 6],
        "l2_leaf_reg": [1, 5],
    },
    "MLP": {
        "activation": ["relu", "tanh"],
        "hidden_layer_sizes": [(10,), (12,), (10, 5), (12, 6)],
        "alpha": [0.05, 0.1, 0.5, 1.0],
        "learning_rate_init": [0.001, 0.003],
    },
}