import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import confusion_matrix

from decision_tree import group_predicted_products
from metrics import split_train_df, calc_avg_f1, kfold_split
from plots import plot_kfold_scores
from sounds import beep



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


def ka_add_groupby_features_1_vs_n(df, group_columns_list, agg_dict):
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

    grouped = df.groupby(group_columns_list)

    the_stats = grouped.agg(agg_dict)
    the_stats.columns = the_stats.columns.droplevel(0)
    the_stats.reset_index(inplace=True)

    return the_stats


def ka_add_groupby_features_n_vs_1(df, group_columns_list, target_columns_list, methods_list):
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

    grouped_name = ''.join(group_columns_list)
    target_name = ''.join(target_columns_list)
    combine_name = [[grouped_name] + [method_name] + [target_name] for method_name in methods_list]

    grouped = df.groupby(group_columns_list)

    the_stats = grouped[target_name].agg(methods_list).reset_index()
    the_stats.columns = [grouped_name] + \
                        ['_%s_%s_by_%s' % (grouped_name, method_name, target_name) \
                         for (grouped_name, method_name, target_name) in combine_name]
    return the_stats


def get_UP_relative_streak_feature():
    streak_df = pd.read_csv('./processed/product_streak.csv')
    streak_df['UP_relative_streak'] = streak_df['product_streak_last_order'] / streak_df['number_of_client_orders']
    return streak_df[['user_id', 'product_id', 'UP_relative_streak']]


def get_product_shares_features(prior_orders):
    res = None
    for area in ['aisle', 'department']:
        area_field = '{}_id'.format(area)
        area_size_filed = '{}_size'.format(area)
        products_orders = prior_orders.groupby([area_field, 'product_id'])['order_id'].count().rename('product_size').reset_index()
        area_orders = prior_orders.groupby([area_field])['order_id'].count().rename(area_size_filed).reset_index()
        shares_df = pd.merge(products_orders, area_orders)
        shares_df['_p_{}_share'.format(area)] = shares_df['product_size'] / shares_df[area_size_filed]
        if res is None:
            res = shares_df
        else:
            res = pd.merge(shares_df, res)

    return res[['product_id', '_p_aisle_share', '_p_department_share']]


def get_user_diversity_features(prior_orders):
    return pd.merge(prior_orders.groupby(['user_id'])['aisle_id'].nunique().rename('_u_aisle_diversity').reset_index(),
                    prior_orders.groupby(['user_id'])['department_id'].nunique().rename('_u_department_diversity').reset_index())


if __name__ == '__main__':
    path_data = 'input/'
    priors, train, orders, products, aisles, departments, sample_submission = load_data(path_data)

    orders['days_since_first_order'] = orders.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)

    # Products information ----------------------------------------------------------------
    # add order information to priors set
    priors_orders_detail = orders.merge(right=priors, how='inner', on='order_id')
    priors_orders_detail = priors_orders_detail.merge(right=products, how='inner', on='product_id')

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
                                                    'add_to_cart_order': {'_up_average_cart_position': 'mean'}})

    data = data.merge(prd, how='inner', on='product_id').merge(users, how='inner', on='user_id')

    data = data.merge(get_UP_relative_streak_feature())
    data = data.merge(get_product_shares_features(priors_orders_detail))
    data = data.merge(get_user_diversity_features(priors_orders_detail))

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
        ['user_id','product_id','product_prior_order_day','avg_interval','interval_std','n_intervals']]
    data = pd.merge(data, product_orders_intervals, on=['user_id', 'product_id'], how='left')

    data['user_product_frequency_indicator'] = (data['n_intervals'] > 5) & (data['avg_interval'] > (4 * data['interval_std'])) & \
                                   ((data['days_since_first_order'] - data['product_prior_order_day']) > (data['avg_interval'] - 2*data['interval_std'])) & \
                                   ((data['days_since_first_order'] - data['product_prior_order_day']) < (data['avg_interval'] + 2*data['interval_std']))


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

    FEATURES = ['_up_average_cart_position',
                '_up_last_order_number',
                '_up_first_order_number',
                '_up_order_count',
                '_prod_reorder_tot_cnts',
                '_prod_tot_cnts',
                '_prod_buy_first_time_total_cnt',
                '_prod_buy_second_time_total_cnt',
                '_prod_reorder_prob',
                '_prod_reorder_ratio',
                '_prod_reorder_times',
                '_user_sum_days_since_prior_order',
                '_user_mean_days_since_prior_order',
                '_user_total_orders',
                '_user_total_products',
                '_user_distinct_products',
                '_user_reorder_ratio',
                '_user_average_basket',
                'time_since_last_order',
                '_up_order_rate',
                '_up_order_since_last_order',
                '_up_order_rate_since_first_order',
                '_p_aisle_share',
                '_p_department_share',
                '_u_aisle_diversity',
                '_u_department_diversity']
                # 'UP_relative_streak' -- no gain

    # todo aisle share/department share
    # todo '_ua_reorder_ration', '_ud_reorder_ration',  product diversity (aisle, department)
    # todo trend features

    TARGET_FEATURE = 'reordered'

    CLASSIFIER_THRESHOLD = 0.2

    eval = False
    kfold_valid = True
    print_confusion_matrix = False

    if eval:
        if kfold_valid:
            scores = []

            for valid_train, valid_test in kfold_split(train, 6):
                d_train = xgboost.DMatrix(valid_train[FEATURES], valid_train[TARGET_FEATURE])

                watchlist = [(d_train, "train")]
                bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)

                d_test = xgboost.DMatrix(valid_test[FEATURES])
                predicted = (bst.predict(d_test) > CLASSIFIER_THRESHOLD)
                valid_test['prediction'] = (predicted | valid_test['user_product_frequency_indicator']).astype(int)

                if print_confusion_matrix:
                    print(confusion_matrix(valid_test[TARGET_FEATURE], valid_test['prediction']))

                avg_f1_score = calc_avg_f1(group_predicted_products(valid_test, TARGET_FEATURE),
                                           group_predicted_products(valid_test, prediction_column='prediction'))
                print(avg_f1_score)
                scores.append(avg_f1_score)

            plot_kfold_scores(scores)
            beep()
        else:
            valid_train, valid_test = split_train_df(train)
            d_train = xgboost.DMatrix(valid_train[FEATURES], valid_train[TARGET_FEATURE])

            watchlist = [(d_train, "train")]
            bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)

            d_test = xgboost.DMatrix(valid_test[FEATURES])

            predicted = (bst.predict(d_test) > CLASSIFIER_THRESHOLD)
            valid_test['prediction'] = (predicted | valid_test['user_product_frequency_indicator']).astype(int)

            if print_confusion_matrix:
                print(confusion_matrix(valid_test[TARGET_FEATURE], valid_test['prediction']))

            avg_f1_score = calc_avg_f1(group_predicted_products(valid_test, TARGET_FEATURE),
                                       group_predicted_products(valid_test, prediction_column='prediction'))
            print(avg_f1_score)
            beep()
    else:
        X_train, y_train = train[FEATURES], train[TARGET_FEATURE]
        d_train = xgboost.DMatrix(X_train, y_train)

        watchlist = [(d_train, "train")]
        bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
        # xgboost.plot_importance(bst)

        d_test = xgboost.DMatrix(X_test[FEATURES])
        X_test.loc[:, TARGET_FEATURE] = ((bst.predict(d_test) > CLASSIFIER_THRESHOLD) | X_test['user_product_frequency_indicator']).astype(int)
        X_test.loc[:, 'product_id'] = X_test.product_id.astype(str)
        submit = ka_add_groupby_features_n_vs_1(X_test[X_test[TARGET_FEATURE] == 1],
                                                group_columns_list=['order_id'],
                                                target_columns_list=['product_id'],
                                                methods_list=[lambda x: ' '.join(set(x))])
        submit.columns = sample_submission.columns.tolist()
        submit_final = sample_submission[['order_id']].merge(submit, how='left').fillna('None')
        submit_final.to_csv("xgboost.csv", index=False)
        beep()