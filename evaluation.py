import numpy as np

from config import RANDOM_SEEDS


def aggregate_results(all_seed_metrics):
    seeds = list(all_seed_metrics.keys())
    algorithms = list(all_seed_metrics[seeds[0]].keys())
    metric_keys = ['r', 'r2', 'mse', 'rmse', 'mae', 'mape']
    aggregated = {}
    for alg in algorithms:
        aggregated[alg] = {}
        for split in ('train', 'test'):
            for m in metric_keys:
                values = [all_seed_metrics[s][alg][split][m] for s in seeds]
                aggregated[alg][f'{split}_{m}_mean'] = float(np.mean(values))
                # ddof=1 for sample std to be consistent with excel's STDEV.S
                aggregated[alg][f'{split}_{m}_std'] = float(np.std(values, ddof=1))
    return aggregated


def find_representative_seed(all_seed_metrics, aggregated):
    seeds = list(all_seed_metrics.keys())
    algorithms = list(aggregated.keys())
    best_alg = min(algorithms, key=lambda a: aggregated[a]['test_rmse_mean'])
    # representative seed: minimizes sum of |RMSE_s,a - mean_RMSE_a| across all models
    rep_seed = min(
        seeds,
        key=lambda s: sum(
            abs(all_seed_metrics[s][a]['test']['rmse'] - aggregated[a]['test_rmse_mean'])
            for a in algorithms if a in all_seed_metrics[s]
        )
    )
    return rep_seed, best_alg


def print_seed_summary(seed, seed_metrics):
    metric_keys = ['r2', 'rmse', 'mae', 'mape']
    metric_labels = ['R2', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE(%)']
    sorted_algs = sorted(seed_metrics.keys(),
                         key=lambda a: seed_metrics[a]['test']['rmse'])
    print(f"\n  Seed {seed} Test Set Results:")
    header = f"{'Method':<6}" + "".join(f"{lbl:>14}" for lbl in metric_labels)
    print(header)
    for alg in sorted_algs:
        row = f"{alg:<6}"
        for mk in metric_keys:
            row += f"{seed_metrics[alg]['test'][mk]:>10.4f}"
        print(row)


def print_summary(aggregated):
    metric_labels = ['R2', 'RMSE (MPa)', 'MAE (MPa)', 'MAPE(%)']
    metric_keys = ['r2', 'rmse', 'mae', 'mape']
    n_seeds = len(RANDOM_SEEDS)
    print("\nRESULTS SUMMARY (mean +/- std across {} seed(s))\n".format(n_seeds))
    sorted_algs = sorted(aggregated.keys(),
                         key=lambda a: aggregated[a]['test_rmse_mean'])
    header = f"{'Method':<6}" + "".join(f"{'Test ' + lbl:>20}" for lbl in metric_labels)
    print(header)
    for alg in sorted_algs:
        row = f"{alg:<6}"
        for mk in metric_keys:
            m = aggregated[alg][f'test_{mk}_mean']
            sd = aggregated[alg][f'test_{mk}_std']
            if n_seeds > 1:
                row += f"{m:6.4f}+/-{sd:.4f}"
            else:
                row += f"{m:>14.4f}"
        print(row)
    print()