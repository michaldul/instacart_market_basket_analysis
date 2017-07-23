# coding: utf-8

import pandas as pd
import numpy as np
from metrics import split_train_df
from sklearn.ensemble import RandomForestClassifier
import metrics


def prepare_train_df():
    all_products_client_ordered = pd.merge(order_products_prior, orders)[['user_id', 'product_id']].drop_duplicates()
    all_train_orders = orders[orders['eval_set'] == 'train'][['user_id', 'order_id']]
    training_skeleton = pd.merge(all_train_orders, all_products_client_ordered)
    print('train skeleton size:', training_skeleton.shape)
    training_df = pd.merge(training_skeleton, order_products_train, how='left')
    training_df['y'] = False
    training_df.ix[training_df['add_to_cart_order'].notnull(), 'y'] = True
    training_df = pd.merge(training_df, product_features, left_on=['product_id'], right_index=True, how='left')
    training_df = pd.merge(training_df, orders)
    training_df = pd.merge(training_df, user_features, left_on=['user_id'], right_index=True)
    training_df = pd.merge(training_df, user_product_features, how='left')
    training_df = training_df.fillna(0)

    return training_df


def prepare_test_df():
    all_products_client_ordered = pd.merge(order_products_prior, orders)[['user_id', 'product_id']].drop_duplicates()
    all_test_orders = orders[orders['eval_set'] == 'test'][['user_id', 'order_id']]
    test_df = pd.merge(all_test_orders, all_products_client_ordered)
    print('test skeleton size:', test_df.shape)

    test_df = pd.merge(test_df, product_features, left_on=['product_id'], right_index=True, how='left')
    test_df = pd.merge(test_df, orders)
    test_df = pd.merge(test_df, user_features, left_on=['user_id'], right_index=True)
    test_df = pd.merge(test_df, user_product_features, how='left')
    test_df = test_df.fillna(0)

    return test_df


def group_predicted_products(df, prediction_column='prediction'):
    """
    >>> (group_predicted_products(pd.DataFrame({'order_id': [1, 1, 2], 'product_id': [1, 2, 3], 'prediction': [False, True, False]})) == pd.Series({1:[2], 2:[]})).all()
    True
    >>> group_predicted_products(pd.DataFrame({'order_id': [1], 'product_id': [1], 'p': [False]}), 'p').name
    'products'
    """
    without_empty_orders = (df[df[prediction_column] == True]
                            .groupby('order_id')['product_id']
                            .apply(lambda x: [int(p) for p in x])
                            .rename('products'))
    all_empty = df.groupby('order_id').apply(lambda _: [])
    combined = pd.concat([without_empty_orders, all_empty], axis=1)

    return combined['products'].combine_first(all_empty)


def generate_submmision(sub_df):
    """
    >>> (generate_submmision(pd.DataFrame({'order_id': [1, 1, 2], 'product_id': [1, 2, 3], 'prediction': [False, True, False]})) == pd.Series({1:'2', 2:'None'})).all()
    True
    """
    return group_predicted_products(sub_df)\
        .apply(lambda x: ' '.join([str(p) for p in x]) if len(x) else 'None')



if __name__ == '__main__':
    aisles = pd.read_csv('input/aisles.csv')
    departments = pd.read_csv('input/departments.csv')
    products = pd.read_csv('input/products.csv')
    order_products_prior = pd.read_csv('input/order_products__prior.csv')
    order_products_train = pd.read_csv('input/order_products__train.csv')
    orders = pd.read_csv('input/orders.csv')

    prior = pd.merge(order_products_prior, products)
    prior = pd.merge(prior, orders)

    product_features = prior.groupby('product_id').agg({'product_id': 'count',
                                                        'reordered': 'sum',
                                                        'add_to_cart_order': 'mean'})
    product_features['reorder_probability'] = product_features['reordered'] / product_features['product_id']
    product_features['product_id'] = np.log(product_features['product_id'])

    product_features.rename(columns={'product_id': 'log_prod_orders',
                                     'reordered': 'log_n_reordered',
                                     'add_to_cart_order': 'n_added'
                                     }, inplace=True)

    user_features = prior.groupby('user_id').agg({'order_id': lambda x: len(np.unique(x)),
                                                  'days_since_prior_order': ['mean', 'sum'],
                                                  'product_id': lambda x: len(np.unique(x)),
                                                  })

    user_features.columns.set_levels(['number_of_distinct_products',
                                      'number_of_orders',
                                      'avg_days_since_order',
                                      'total_days_active'],
                                     level=1, inplace=True)
    user_features.columns = user_features.columns.droplevel()

    user_product_features = prior.groupby(['user_id', 'product_id']).agg({'order_id': 'count'}).rename(
        columns={'order_id': 'number_of_product_orders'})

    user_prod_prop = pd.merge(user_product_features.reset_index(), user_features, left_on=['user_id'], right_index=True)
    user_prod_prop['prop'] = user_prod_prop['number_of_product_orders'] / (user_prod_prop['number_of_orders'] + 1)
    user_product_features['prop_of_order'] = user_prod_prop.set_index(['user_id', 'product_id'])['prop']

    user_product_features.reset_index(inplace=True)

    training_df = prepare_train_df()

    train_features = ['log_prod_orders', 'log_n_reordered', 'n_added',
                      'reorder_probability', 'number_of_distinct_products', 'number_of_distinct_products',
                      'number_of_orders', 'avg_days_since_order', 'number_of_product_orders',
                      'prop_of_order']

    eval = True

    if eval:
        train, test = split_train_df(training_df)

        clf = RandomForestClassifier()
        clf.fit(train[train_features], train['y'])

        predicted = clf.predict(test[train_features])

        sub_df = test[['order_id', 'product_id', 'y']]

        sub_df['pred'] = predicted

        print(metrics.calc_avg_f1(group_predicted_products(sub_df, 'y'), group_predicted_products(sub_df, 'pred')))
    else:
        test_df = prepare_test_df()

        clf = RandomForestClassifier()
        clf.fit(training_df[train_features], training_df['y'])

        test_df['prediction'] = clf.predict(test_df[train_features])

        generate_submmision(test_df[['order_id', 'product_id', 'prediction']]) \
            .to_csv('decision_tree.csv', header=['products'], index_label=['order_id'])


# todo: last order info
