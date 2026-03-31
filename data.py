import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    SELECTED_FEATURES, CATEGORICAL_FEATURES, TARGET, TEST_SIZE
)

def load_data(filepath):
    df = pd.read_excel(filepath)
    print(f"Dataset loaded: {df.shape[0]} specimens, {df.shape[1]} columns.")
    return df


def prepare_splits(df, seed):
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET].copy()
    X_raw = X.copy()

    # Dummy encoding with first category dropped: MLR, SVR
    X_drop_first = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Full one-hot encoding: KNN, DT, RF, GB, XGB, MLP
    X_onehot = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Native category dtype: LGBM
    X_lgb = X.copy()
    for col in CATEGORICAL_FEATURES:
        X_lgb[col] = X_lgb[col].astype('category')

    # String categories: CB
    X_cb = X.copy()
    for col in CATEGORICAL_FEATURES:
        X_cb[col] = X_cb[col].astype(str)

    # Split is performed on X_drop_first 
    # Indices are reused across all encodings
    # to guarantee identical train/test partitions for all models

    # 80/20 split
    X_tr_df, X_te_df, y_train, y_test = train_test_split(
        X_drop_first, y, test_size=TEST_SIZE, random_state=seed
    )
    train_idx = X_tr_df.index
    test_idx = X_te_df.index

    return {
        'X_train': X_tr_df,
        'X_test': X_te_df,
        'X_train_oh': X_onehot.loc[train_idx],
        'X_test_oh': X_onehot.loc[test_idx],
        'X_train_lgb': X_lgb.loc[train_idx],
        'X_test_lgb': X_lgb.loc[test_idx],
        'X_train_cb': X_cb.loc[train_idx],
        'X_test_cb': X_cb.loc[test_idx],
        'y_train': y_train,
        'y_test': y_test,
        'X_raw_train': X_raw.loc[train_idx],
        'X_raw_test': X_raw.loc[test_idx],
        'specimen_train': df.loc[train_idx, 'specimen'],
        'specimen_test': df.loc[test_idx, 'specimen'],
    }