import pandas as pd
import xgboost
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt

from plots import plot_kfold_scores
from sounds import beep

from base_model import load_data, ka_add_groupby_features_1_vs_n, get_UP_relative_streak_feature

TARGET_FEATURE = 'reordered'

if __name__ == '__main__':
    path_data = 'input/'
    priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data)

    orders['days_since_first_order'] = orders.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)

    priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')

    priors_orders_detail.loc[:, '_user_buy_product_times'] = \
        priors_orders_detail.groupby(['user_id', 'product_id']).cumcount() + 1

    prd = ka_add_groupby_features_1_vs_n(priors_orders_detail, ['product_id'],
                                         {'user_id': {'_prod_tot_cnts': 'count'},
                                          'reordered': {'_prod_reorder_tot_cnts': 'sum'}})

    prd = prd.set_index('product_id')
    prd['_prod_buy_first_time_total_cnt'] = priors_orders_detail[priors_orders_detail['_user_buy_product_times'] == 1] \
        .groupby('product_id')['_user_buy_product_times'].count()
    prd['_prod_buy_second_time_total_cnt'] = priors_orders_detail[priors_orders_detail['_user_buy_product_times'] == 2] \
        .groupby('product_id')['_user_buy_product_times'].count()
    prd = prd.fillna(0)
    prd = prd.reset_index()

    prd['_prod_reorder_prob'] = prd._prod_buy_second_time_total_cnt / prd._prod_buy_first_time_total_cnt
    prd['_prod_reorder_ratio'] = prd._prod_reorder_tot_cnts / prd._prod_tot_cnts
    prd['_prod_reorder_times'] = 1 + prd._prod_reorder_tot_cnts / prd._prod_buy_first_time_total_cnt

    users = ka_add_groupby_features_1_vs_n(orders[orders.eval_set == 'prior'], ['user_id'],
                                           {'order_number': {'_user_total_orders': 'max'},
                                            'days_since_prior_order': {
                                                '_user_sum_days_since_prior_order': 'sum',
                                                '_user_mean_days_since_prior_order': 'mean'}}
                                           )

    users = pd.merge(users, orders[orders.eval_set != 'prior'][['user_id', 'days_since_first_order']])

    us = pd.concat([
        priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
        priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
        (priors_orders_detail.groupby('user_id')['reordered'].sum() /
         priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')[
             'order_number'].count()).rename(
            '_user_reorder_ratio')
    ], axis=1).reset_index()

    users = users.merge(us, how='inner')

    users['_user_average_basket'] = users._user_total_products / users._user_total_orders

    us = orders[orders.eval_set != "prior"][['user_id', 'order_id', 'eval_set', 'days_since_prior_order']]
    us.rename(index=str, columns={'days_since_prior_order': 'time_since_last_order'}, inplace=True)

    users = users.merge(us, how='inner')

    data = ka_add_groupby_features_1_vs_n(df=priors_orders_detail,
                                          group_columns_list=['user_id', 'product_id'],
                                          agg_dict={'order_number': {'_up_order_count': 'count',
                                                                     '_up_first_order_number': 'min',
                                                                     '_up_last_order_number': 'max'},
                                                    'add_to_cart_order': {'_up_average_cart_position': 'mean'}})

    data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')

    data = data.merge(get_UP_relative_streak_feature())

    data['_up_order_rate'] = data._up_order_count / data._user_total_orders
    data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
    data['_up_order_rate_since_first_order'] = data._up_order_count / (
        data._user_total_orders - data._up_first_order_number + 1)

    # add user_id to train set
    train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
    # better approach than "skeleton-building"
    data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

    # add features related to frequency of buying product by the client
    product_orders_intervals = pd.read_csv('processed/product_orders_intervals.csv')[
        ['user_id', 'product_id', 'product_prior_order_day', 'avg_interval', 'interval_std', 'n_intervals']]
    data = pd.merge(data, product_orders_intervals, on=['user_id', 'product_id'], how='left')

    data['user_product_frequency_indicator'] = (data['n_intervals'] > 5) & (
    data['avg_interval'] > (4 * data['interval_std'])) & \
                                               ((data['days_since_first_order'] - data['product_prior_order_day']) > (
                                               data['avg_interval'] - 2 * data['interval_std'])) & \
                                               ((data['days_since_first_order'] - data['product_prior_order_day']) < (
                                               data['avg_interval'] + 2 * data['interval_std']))

    train = data.loc[data.eval_set == "train", :]
    train.loc[:, 'reordered'] = train.reordered.fillna(0)

    X_test = data.loc[data.eval_set == "test", :]

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "logloss"
        , "eta": 0.1
        , "max_depth": 6
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
        # , 'updater': 'grow_gpu'

    }

    FEATURES = ['_up_average_cart_position', '_up_last_order_number',
                '_up_first_order_number', '_up_order_count', '_prod_reorder_tot_cnts',
                '_prod_tot_cnts', '_prod_buy_first_time_total_cnt',
                '_prod_buy_second_time_total_cnt', '_prod_reorder_prob',
                '_prod_reorder_ratio', '_prod_reorder_times',
                '_user_sum_days_since_prior_order', '_user_mean_days_since_prior_order',
                '_user_total_orders', '_user_total_products', '_user_distinct_products',
                '_user_reorder_ratio', '_user_average_basket', 'time_since_last_order',
                '_up_order_rate', '_up_order_since_last_order',
                '_up_order_rate_since_first_order']  # , 'UP_relative_streak']

    train = train.groupby('order_id').agg({'_up_average_cart_position': 'mean',
                                           '_up_last_order_number': 'mean',
                                           '_up_first_order_number': 'mean',
                                           '_up_order_count': 'mean',
                                           '_prod_reorder_tot_cnts': 'mean',
                                           '_prod_tot_cnts': 'mean',
                                           '_prod_buy_first_time_total_cnt': 'mean',
                                           '_prod_buy_second_time_total_cnt': 'mean',
                                           '_prod_reorder_prob': 'mean',
                                           '_prod_reorder_ratio': 'mean',
                                           '_prod_reorder_times': 'mean',
                                           '_user_sum_days_since_prior_order': 'max',
                                           '_user_mean_days_since_prior_order': 'max',
                                           '_user_total_orders': 'max',
                                           '_user_total_products': 'max',
                                           '_user_distinct_products': 'max',
                                           '_user_reorder_ratio': 'max',
                                           '_user_average_basket': 'max',
                                           'time_since_last_order': 'max',
                                           '_up_order_rate': 'mean',
                                           '_up_order_since_last_order': 'mean',
                                           '_up_order_rate_since_first_order': 'mean',
                                           'reordered': 'sum'
                                           })

    train[TARGET_FEATURE] = train[TARGET_FEATURE] == 0

    CLASSIFIER_THRESHOLD = 0.21

    eval = True
    kfold_valid = True
    print_confusion_matrix = False

    if eval:
        if kfold_valid:
            scores = []

            for valid_train_idx, valid_test_idx in KFold(n_splits=5).split(train):
                valid_train = train.iloc[valid_train_idx]
                valid_test = train.iloc[valid_test_idx]
                d_train = xgboost.DMatrix(valid_train[FEATURES], valid_train[TARGET_FEATURE])

                watchlist = [(d_train, "train")]
                bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist,
                                    verbose_eval=10)

                d_test = xgboost.DMatrix(valid_test[FEATURES])

                predicted = (bst.predict(d_test) > CLASSIFIER_THRESHOLD)
                valid_test['prediction'] = predicted.astype(int)

                f1 = f1_score(valid_test[TARGET_FEATURE], valid_test['prediction'])
                print(f1)
                scores.append(f1)

                print(confusion_matrix(valid_test[TARGET_FEATURE], valid_test['prediction']))

            plot_kfold_scores(scores)
            beep()
    else:
        X_train, y_train = train[FEATURES], train[TARGET_FEATURE]
        d_train = xgboost.DMatrix(X_train, y_train)

        watchlist = [(d_train, 'train')]
        bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)

        d_test = xgboost.DMatrix(X_test[FEATURES])
        X_test['none'] = (bst.predict(d_test) > CLASSIFIER_THRESHOLD).astype(int)
        X_test[['order_id', 'none']].to_csv('none_prediction.csv', index=False)