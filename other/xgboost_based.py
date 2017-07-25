import gc
import time
import numpy as np
import pandas as pd
import xgboost

import matplotlib.pyplot as plt
from decision_tree import group_predicted_products
from metrics import split_train_df, calc_avg_f1


def load_data(path_data):
    """
    --------------------------------order_product--------------------------------
    * Unique in order_id + product_id
    """
    priors = pd.read_csv(path_data + 'order_products__prior.csv',
                         dtype={
                             'order_id': np.int32,
                             'product_id': np.uint16,
                             'add_to_cart_order': np.int16,
                             'reordered': np.int8})
    train = pd.read_csv(path_data + 'order_products__train.csv',
                        dtype={
                            'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})
    '''
    --------------------------------order--------------------------------
    * This file tells us which set (prior, train, test) an order belongs
    * Unique in order_id
    * order_id in train, prior, test has no intersection
    * this is the #order_number order of this user
    '''
    orders = pd.read_csv(path_data + 'orders.csv',
                         dtype={
                             'order_id': np.int32,
                             'user_id': np.int64,
                             'eval_set': 'category',
                             'order_number': np.int16,
                             'order_dow': np.int8,
                             'order_hour_of_day': np.int8,
                             'days_since_prior_order': np.float32})

    '''
    --------------------------------product--------------------------------
    * Unique in product_id
    '''
    products = pd.read_csv(path_data + 'products.csv')
    aisles = pd.read_csv(path_data + "aisles.csv")
    departments = pd.read_csv(path_data + "departments.csv")
    sample_submission = pd.read_csv(path_data + "sample_submission.csv")

    return priors, train, orders, products, aisles, departments, sample_submission


class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            end_time = time.time()
            print(self.process_name + " end ......")
            print('time lapsing {0} s \n'.format(end_time - self.begin_time))


def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict, only_new_feature=True):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features"):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if only_new_feature:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new


def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list, keep_only_stats=True,
                                   verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [1 column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element
       methods_list: list_like
          methods that you want to use, all methods that supported by groupby in Pandas

       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_stats_features_n_vs_1(train, group_columns_list=['x0'], target_columns_list=['x10'])
    '''
    with tick_tock("add stats features", verbose):
        dicts = {"group_columns_list": group_columns_list, "target_columns_list": target_columns_list,
                 "methods_list": methods_list}

        for k, v in dicts.items():
            try:
                if type(v) == list:
                    pass
                else:
                    raise TypeError(k + "should be a list")
            except TypeError as e:
                print(e)
                raise

        grouped_name = ''.join(group_columns_list)
        target_name = ''.join(target_columns_list)
        combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped[target_name].agg(methods_list).reset_index()
        the_stats.columns = [grouped_name] + \
                            ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                             for (grouped_name, method_name, target_name) in combine_name]
        if keep_only_stats:
            return the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')
        return df_new


path_data = 'input/'
priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data)

# Products information ----------------------------------------------------------------
# add order information to priors set
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

us = pd.concat([
    priors_orders_detail.groupby('user_id')['product_id'].count().rename('_user_total_products'),
    priors_orders_detail.groupby('user_id')['product_id'].nunique().rename('_user_distinct_products'),
    (priors_orders_detail.groupby('user_id')['reordered'].sum() /
     priors_orders_detail[priors_orders_detail['order_number'] > 1].groupby('user_id')['order_number'].count()).rename(
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
                                                'add_to_cart_order': {'_up_average_cart_position': 'mean'}}
                                      )

data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')

data['_up_order_rate'] = data._up_order_count / data._user_total_orders
data['_up_order_since_last_order'] = data._user_total_orders - data._up_last_order_number
data['_up_order_rate_since_first_order'] = data._up_order_count / (
    data._user_total_orders - data._up_first_order_number + 1)

# add user_id to train set
train = train.merge(right=orders[['order_id', 'user_id']], how='left', on='order_id')
# better approach than "skeleton-building"
data = data.merge(train[['user_id', 'product_id', 'reordered']], on=['user_id', 'product_id'], how='left')

train = data.loc[data.eval_set == "train", :]
train.loc[:, 'reordered'] = train.reordered.fillna(0)

del priors_orders_detail, orders
gc.collect()

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
            '_up_order_rate_since_first_order']

eval = False


if eval:
    tresholds = []
    scores = []
    for i in range(1):
        train, valid = split_train_df(train)
        d_train = xgboost.DMatrix(train[FEATURES], train.reordered)

        watchlist = [(d_train, "train")]
        bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
        # xgboost.plot_importance(bst)

        d_test = xgboost.DMatrix(valid.drop(['eval_set', 'user_id', 'product_id', 'order_id', 'reordered'], axis=1))
        treshold = 0.19 + i * 0.01
        valid['prediction'] = (bst.predict(d_test) > treshold).astype(int)

        score = calc_avg_f1(group_predicted_products(valid, 'reordered'),
                            group_predicted_products(valid, prediction_column='prediction'))
        tresholds.append(treshold)
        scores.append(score)

        print(scores)
    if len(scores) > 1:
        plt.plot(tresholds, scores)
        plt.show()
else:
    X_test = data.loc[data.eval_set == "test", :]

    X_train, y_train = train[FEATURES], train.reordered

    d_train = xgboost.DMatrix(X_train, y_train)

    watchlist = [(d_train, "train")]
    bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
    xgboost.plot_importance(bst)

    d_test = xgboost.DMatrix(X_test[FEATURES])
    X_test.loc[:, 'reordered'] = (bst.predict(d_test) > 0.2).astype(int)
    X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
    submit = ka_add_groupby_features_n_vs_1(X_test[X_test.reordered == 1],
                                            group_columns_list=['order_id'],
                                            target_columns_list=['product_id'],
                                            methods_list=[lambda x: ' '.join(set(x))], keep_only_stats=True)
    submit.columns = sample_submission.columns.tolist()
    submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
    submit_final.to_csv("xgboost.csv", index=False)
