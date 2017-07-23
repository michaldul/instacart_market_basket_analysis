import pandas as pd
from sklearn.model_selection import train_test_split


def calc_f1(y_true, y_pred):
    """
    >>> np.fabs(calc_f1([1, 2, 3], [1, 3]) - 0.8) < 0.0000001
    True
    >>> calc_f1([], [])
    1
    """
    y_true, y_pred = set(y_true), set(y_pred)

    tp = len(y_true.intersection(y_pred))
    fn = len([p for p in y_true if p not in y_pred])
    fp = len([p for p in y_pred if p not in y_true])

    if fn + fp == 0:
        return 1
    return 2 * tp / (2 * tp + fn + fp)


def calc_avg_f1(y_true : pd.Series, y_pred : pd.Series):
    f1_df = pd.concat([y_true.rename('y_true'), y_pred.rename('y_pred')], axis=1)
    f1_df['y_true'].loc[f1_df['y_true'].isnull()] = f1_df['y_true'].loc[f1_df['y_true'].isnull()].apply(lambda x: [])
    f1_df['y_pred'].loc[f1_df['y_pred'].isnull()] = f1_df['y_pred'].loc[f1_df['y_pred'].isnull()].apply(lambda x: [])

    return f1_df.apply(lambda r: calc_f1(r['y_true'], r['y_pred']), axis=1).mean()


def split_train_df(train_df):
    orders = train_df['order_id'].unique()
    train_orders, test_orders = train_test_split(orders, test_size=0.3)
    train = train_df[train_df['order_id'].isin(train_orders)]
    test = train_df[train_df['order_id'].isin(test_orders)]

    return train, test