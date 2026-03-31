# Code for the manuscript:
# "Machine learning based prediction of joint shear strength
# for fiber reinforced concrete beam-column connections"
# by Yunus Kantekin, Burcu Burak Bakir and Rodrigo Sarlo

import sys
import os
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

import joblib

from config import (
    RANDOM_SEEDS, TEST_SIZE, SELECTED_FEATURES,
    TARGET, DATA_FILE, CATEGORICAL_FEATURES,
)
from data import load_data, prepare_splits
from models import train_one_seed
from evaluation import (
    aggregate_results, find_representative_seed,
    print_seed_summary, print_summary,
)
from shap_analysis import run_shap_analysis, run_shap_comparison
from export import export_to_excel

_MODEL_FULL_NAMES = {
    'MLR': 'Multiple Linear Regression',
    'KNN': 'K-Nearest Neighbors',
    'SVR': 'Support Vector Regression',
    'DT': 'Decision Tree',
    'RF': 'Random Forest',
    'GB': 'Gradient Boosting',
    'XGB': 'XGBoost',
    'LGBM': 'LightGBM',
    'CB': 'CatBoost',
    'MLP': 'Multi-Layer Perceptron',
}


def main():
    print("ML based prediction of vj for FRC beam-column joints\n")
    print(f"Number of random seeds used: {len(RANDOM_SEEDS)}")
    print(f"Test split used: {int(TEST_SIZE * 100)}%")
    print(f"Number of features: {len(SELECTED_FEATURES)}")
    print("Target: joint shear strength")
    print()

    data_path = os.path.join(os.path.dirname(__file__), DATA_FILE)
    df = load_data(data_path)

    all_seed_metrics = {}
    all_seed_models = {}
    all_seed_splits = {}
    all_seed_preds = {}
    all_seed_best_params = {}

    for seed in RANDOM_SEEDS:
        print(f"\nRunning Seed {seed}:")
        splits = prepare_splits(df, seed)
        print(
            f"Train: {len(splits['X_train'])} specimens & "
            f"Test: {len(splits['X_test'])} specimens"
        )
        seed_metrics, seed_models, seed_preds, seed_best_params = train_one_seed(splits, seed)
        all_seed_metrics[seed] = seed_metrics
        all_seed_models[seed] = seed_models
        all_seed_splits[seed] = splits
        all_seed_preds[seed] = seed_preds
        all_seed_best_params[seed] = seed_best_params
        print_seed_summary(seed, seed_metrics)

    aggregated = aggregate_results(all_seed_metrics)

    rep_seed, best_alg = find_representative_seed(all_seed_metrics, aggregated)
    print(f"\nBest algorithm based on mean test RMSE: {best_alg}.")
    print(f"Representative seed: {rep_seed}")

    print_summary(aggregated)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(os.path.dirname(__file__), f"Export_{timestamp}")
    os.makedirs(export_dir)
    print(f"\nExport folder created: {os.path.abspath(export_dir)}")
    output_prefix = os.path.join(export_dir, f"ML_Detailed_Results_{timestamp}")
    output_path = output_prefix + ".xlsx"

    shap_df, shap_alg = run_shap_analysis(
        all_seed_models[rep_seed], all_seed_splits[rep_seed],
        aggregated, output_prefix, timestamp
    )
    run_shap_comparison(
        all_seed_models[rep_seed], all_seed_splits[rep_seed],
        output_prefix, timestamp, primary_alg=shap_alg
    )

    export_to_excel(
        aggregated, all_seed_metrics, all_seed_splits, all_seed_preds,
        all_seed_best_params[rep_seed], rep_seed, output_path,
        shap_df, shap_alg
    )

    gui_path = os.path.join(export_dir, 'FRC_BCJ_Rep_ML_Models.joblib')
    # two X_train encodings saved so the GUI can route each model correctly
    joblib.dump({
        'models': {_MODEL_FULL_NAMES[k]: v for k, v in all_seed_models[rep_seed].items()},
        'X_train': all_seed_splits[rep_seed]['X_train'],
        'X_train_oh': all_seed_splits[rep_seed]['X_train_oh'],
        'selected_features': SELECTED_FEATURES,
        'categorical_cols': CATEGORICAL_FEATURES,
        'feature_ranges': {
            feat: {'min': float(df[feat].min()), 'max': float(df[feat].max())}
            for feat in SELECTED_FEATURES if feat not in CATEGORICAL_FEATURES
        },
        'aggregated': aggregated,
        'best_alg': best_alg,
        'rep_seed': rep_seed,
    }, gui_path)
    print(f"\nSaved at: {gui_path}.")

    best_rmse = aggregated[best_alg]['test_rmse_mean']
    print(f"Best model: {best_alg} with mean RMSE={best_rmse:.2f} MPa w/ representative seed: {rep_seed}")


if __name__ == "__main__":
    main()
