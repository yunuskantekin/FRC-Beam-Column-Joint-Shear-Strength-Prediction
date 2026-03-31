import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import PARAM_GRIDS, CATEGORICAL_FEATURES, USE_GPU


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sqrt(max(0.0, r2))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return {
        'r': r, 'r2': r2,
        'mse': mse, 'rmse': rmse,
        'mae': mae, 'mape': mape,
    }


def _pipeline_grid(algorithm_name):
    return {f"model__{k}": v for k, v in PARAM_GRIDS[algorithm_name].items()}


def _strip_prefix(params, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in params.items()}


def train_one_seed(splits, seed):
    X_train = splits['X_train']
    X_test = splits['X_test']
    X_train_oh = splits['X_train_oh']
    X_test_oh = splits['X_test_oh']
    X_train_lgb = splits['X_train_lgb']
    X_test_lgb = splits['X_test_lgb']
    X_train_cb = splits['X_train_cb']
    X_test_cb = splits['X_test_cb']
    y_train = splits['y_train']
    y_test = splits['y_test']

    metrics = {}
    models = {}
    preds = {}
    best_params = {}

    # [1/10] MLR: no hyperparameters to tune: just direct fit
    print("Method 1/10: MLR")
    lr_pipe = LinearRegression()
    lr_pipe.fit(X_train, y_train)
    models["MLR"] = lr_pipe
    best_params["MLR"] = {}
    _tr = lr_pipe.predict(X_train)
    _te = lr_pipe.predict(X_test)
    metrics["MLR"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["MLR"] = {'train': _tr, 'test': _te}

    # [2/10] KNN
    print("Method 2/10: KNN")
    knn_pipe = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())])
    knn_grid = GridSearchCV(
        knn_pipe, param_grid=_pipeline_grid("KNN"),
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    knn_grid.fit(X_train_oh, y_train)
    models["KNN"] = knn_grid.best_estimator_
    best_params["KNN"] = _strip_prefix(knn_grid.best_params_, "model__")
    _tr = knn_grid.best_estimator_.predict(X_train_oh)
    _te = knn_grid.best_estimator_.predict(X_test_oh)
    metrics["KNN"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["KNN"] = {'train': _tr, 'test': _te}

    # [3/10] SVR
    print("Method 3/10: SVR")
    svr_pipe = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
    svr_grid = GridSearchCV(
        svr_pipe, param_grid=_pipeline_grid("SVR"),
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    svr_grid.fit(X_train, y_train)
    models["SVR"] = svr_grid.best_estimator_
    best_params["SVR"] = _strip_prefix(svr_grid.best_params_, "model__")
    _tr = svr_grid.best_estimator_.predict(X_train)
    _te = svr_grid.best_estimator_.predict(X_test)
    metrics["SVR"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["SVR"] = {'train': _tr, 'test': _te}

    # [4/10] DT
    print("Method 4/10: DT")
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=seed),
        param_grid=PARAM_GRIDS["DT"],
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    dt_grid.fit(X_train_oh, y_train)
    models["DT"] = dt_grid.best_estimator_
    best_params["DT"] = dt_grid.best_params_
    _tr = dt_grid.best_estimator_.predict(X_train_oh)
    _te = dt_grid.best_estimator_.predict(X_test_oh)
    metrics["DT"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["DT"] = {'train': _tr, 'test': _te}

    # [5/10] RF
    print("Method 5/10: RF")
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=seed),
        param_grid=PARAM_GRIDS["RF"],
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    rf_grid.fit(X_train_oh, y_train)
    models["RF"] = rf_grid.best_estimator_
    best_params["RF"] = rf_grid.best_params_
    _tr = rf_grid.best_estimator_.predict(X_train_oh)
    _te = rf_grid.best_estimator_.predict(X_test_oh)
    metrics["RF"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["RF"] = {'train': _tr, 'test': _te}

    # [6/10] GB
    print("Method 6/10: GB")
    gb_grid = GridSearchCV(
        GradientBoostingRegressor(random_state=seed),
        param_grid=PARAM_GRIDS["GB"],
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    gb_grid.fit(X_train_oh, y_train)
    models["GB"] = gb_grid.best_estimator_
    best_params["GB"] = gb_grid.best_params_
    _tr = gb_grid.best_estimator_.predict(X_train_oh)
    _te = gb_grid.best_estimator_.predict(X_test_oh)
    metrics["GB"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["GB"] = {'train': _tr, 'test': _te}

    # [7/10] XGB
    print("Method 7/10: XGB")
    try:
        import xgboost as xgb
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=seed, objective="reg:squarederror"),
            param_grid=PARAM_GRIDS["XGB"],
            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        xgb_grid.fit(X_train_oh, y_train)
        models["XGB"] = xgb_grid.best_estimator_
        best_params["XGB"] = xgb_grid.best_params_
        _tr = xgb_grid.best_estimator_.predict(X_train_oh)
        _te = xgb_grid.best_estimator_.predict(X_test_oh)
        metrics["XGB"] = {
            'train': compute_metrics(y_train, _tr),
            'test': compute_metrics(y_test, _te),
        }
        preds["XGB"] = {'train': _tr, 'test': _te}
    except ImportError:
        print(" XGBoost not installed, skipping.")

    # [8/10] LGBM
    print("Method 8/10: LGBM")
    try:
        import lightgbm as lgb
        lgb_grid = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=seed, verbose=-1),
            param_distributions=PARAM_GRIDS["LGBM"],
            n_iter=20, cv=5, scoring='neg_mean_absolute_error',
            n_jobs=1, random_state=seed
        )
        lgb_grid.fit(X_train_lgb, y_train)
        models["LGBM"] = lgb_grid.best_estimator_
        best_params["LGBM"] = lgb_grid.best_params_
        _tr = lgb_grid.best_estimator_.predict(X_train_lgb)
        _te = lgb_grid.best_estimator_.predict(X_test_lgb)
        metrics["LGBM"] = {
            'train': compute_metrics(y_train, _tr),
            'test': compute_metrics(y_test, _te),
        }
        preds["LGBM"] = {'train': _tr, 'test': _te}
    except ImportError:
        print(" LightGBM not installed, skipping.")

    # [9/10] CB
    print("Method 9/10: CB")
    try:
        import catboost as cb_lib
        cat_features = [col for col in X_train_cb.columns if col in CATEGORICAL_FEATURES]
        cb_grid = RandomizedSearchCV(
            cb_lib.CatBoostRegressor(
                random_state=seed, verbose=0, allow_writing_files=False,
                task_type='GPU' if USE_GPU else 'CPU'
            ),
            param_distributions=PARAM_GRIDS["CB"],
            n_iter=5, cv=5, scoring='neg_mean_absolute_error',
            n_jobs=1, random_state=seed
        )
        cb_grid.fit(X_train_cb, y_train,
                    cat_features=cat_features if cat_features else None)
        models["CB"] = cb_grid.best_estimator_
        best_params["CB"] = cb_grid.best_params_
        _tr = cb_grid.best_estimator_.predict(X_train_cb)
        _te = cb_grid.best_estimator_.predict(X_test_cb)
        metrics["CB"] = {
            'train': compute_metrics(y_train, _tr),
            'test': compute_metrics(y_test, _te),
        }
        preds["CB"] = {'train': _tr, 'test': _te}
    except ImportError:
        print("CatBoost not installed, skipping.")

    # [10/10] MLP
    print("Method 10/10: MLP")
    nn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(random_state=seed, verbose=False, max_iter=3000)),
    ])
    # MLP needs the target scaled too, not just the inputs
    ttr = TransformedTargetRegressor(regressor=nn_pipe, transformer=StandardScaler())
    nn_param_grid = {
        f"regressor__model__{k}": v for k, v in PARAM_GRIDS["MLP"].items()
    }
    nn_grid = GridSearchCV(
        ttr, param_grid=nn_param_grid,
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    nn_grid.fit(X_train_oh, y_train)
    models["MLP"] = nn_grid.best_estimator_
    best_params["MLP"] = _strip_prefix(nn_grid.best_params_, "regressor__model__")
    _tr = nn_grid.best_estimator_.predict(X_train_oh)
    _te = nn_grid.best_estimator_.predict(X_test_oh)
    metrics["MLP"] = {
        'train': compute_metrics(y_train, _tr),
        'test': compute_metrics(y_test, _te),
    }
    preds["MLP"] = {'train': _tr, 'test': _te}

    return metrics, models, preds, best_params