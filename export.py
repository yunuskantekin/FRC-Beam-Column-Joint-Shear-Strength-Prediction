import pandas as pd

from config import RANDOM_SEEDS, SELECTED_FEATURES


def export_to_excel(aggregated, all_seed_metrics, all_seed_splits, all_seed_preds,
                    best_params_rep, rep_seed, output_path,
                    shap_df=None, shap_alg=None):

    metric_labels = ['R', 'R2', 'MSE', 'RMSE', 'MAE', 'MAPE (%)']
    metric_keys = ['r', 'r2', 'mse', 'rmse', 'mae', 'mape']
    n_seeds = len(RANDOM_SEEDS)
    sorted_seeds = sorted(all_seed_metrics.keys())
    sorted_algs = sorted(aggregated.keys(),
                         key=lambda a: aggregated[a]['test_rmse_mean'])

    def build_aggregate_df(split):
        rows = []
        for alg in sorted_algs:
            row = {'Algorithm': alg}
            for lbl, mk in zip(metric_labels, metric_keys):
                m = aggregated[alg][f'{split}_{mk}_mean']
                sd = aggregated[alg][f'{split}_{mk}_std']
                if n_seeds > 1:
                    row[f'{lbl} Mean'] = round(m, 6)
                    row[f'{lbl} Std'] = round(sd, 6)
                else:
                    row[lbl] = round(m, 6)
            rows.append(row)
        return pd.DataFrame(rows)

    def build_per_seed_df(split):
        rows = []
        for seed in sorted_seeds:
            for alg in sorted_algs:
                row = {'Seed': seed, 'Algorithm': alg}
                for lbl, mk in zip(metric_labels, metric_keys):
                    row[lbl] = round(all_seed_metrics[seed][alg][split][mk], 6)
                rows.append(row)
        return pd.DataFrame(rows)

    def build_prediction_sheet(split):
        rows = []
        for seed in sorted_seeds:
            sp = all_seed_splits[seed]
            idx_list = sp[f'X_raw_{split}'].index
            specimen = sp[f'specimen_{split}']
            X_raw = sp[f'X_raw_{split}']
            y_true = sp[f'y_{split}']
            alg_pred_series = {}
            for alg in sorted_algs:
                if alg in all_seed_preds[seed]:
                    arr = all_seed_preds[seed][alg][split]
                    alg_pred_series[alg] = pd.Series(arr, index=idx_list)
            for df_idx in idx_list:
                row = {'Seed': seed, 'specimen': specimen.loc[df_idx]}
                for feat in SELECTED_FEATURES:
                    row[feat] = X_raw.loc[df_idx, feat]
                row['vj_exp'] = float(y_true.loc[df_idx])
                for alg, series in alg_pred_series.items():
                    row[alg] = round(float(series[df_idx]), 4)
                rows.append(row)
        return pd.DataFrame(rows)

    model_order = ['MLR', 'KNN', 'SVR', 'DT', 'RF', 'GB', 'XGB', 'LGBM', 'CB', 'MLP']

    def build_hyperparams_df():
        rows = []
        for alg in model_order:
            params = best_params_rep.get(alg, {})
            if not params:
                rows.append({'Model': alg, 'Parameter': '-', 'Value': '(no tuning)'})
            else:
                for param, value in params.items():
                    rows.append({'Model': alg, 'Parameter': param, 'Value': value})
        return pd.DataFrame(rows)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        build_aggregate_df('test').to_excel(
            writer, sheet_name='Test Mean+Std', index=False)
        build_aggregate_df('train').to_excel(
            writer, sheet_name='Train Mean+Std', index=False)
        build_per_seed_df('test').to_excel(
            writer, sheet_name='Test Per Seed', index=False)
        build_per_seed_df('train').to_excel(
            writer, sheet_name='Train Per Seed', index=False)
        build_prediction_sheet('train').to_excel(
            writer, sheet_name='Train Predictions', index=False)
        build_prediction_sheet('test').to_excel(
            writer, sheet_name='Test Predictions', index=False)
        build_hyperparams_df().to_excel(
            writer, sheet_name=f'Representative Hyperparameters (Seed {rep_seed})', index=False
        )
        if shap_df is not None:
            shap_summary = (shap_df.abs().mean()
                            .sort_values(ascending=False)
                            .reset_index())
            shap_summary.columns = ['Feature', 'mean(|SHAP value|)']
            shap_summary.to_excel(
                writer, sheet_name=f'SHAP Summary ({shap_alg})', index=False
            )

    print(f"\nResults exported to: {output_path}")