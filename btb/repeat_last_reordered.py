# Author: Michal Dul
# local 0.2471660010744334
# LB    0.3276826

import pandas as pd

from metrics import calc_avg_f1

eval = True


def get_predictions(X):
    X['prev_order'] = X['order_number'] - 1
    last_orders = pd.merge(orders, X,
                           left_on=['user_id', 'order_number'],
                           right_on=['user_id', 'prev_order'],
                           suffixes=('', '_test'),
                           how='inner')

    prior_reordered = orders_product_prior[orders_product_prior['reordered'] == 1]

    predictions = pd.merge(last_orders, prior_reordered, on=['order_id'], how='left') \
        .groupby('order_id_test')['product_id'].apply(lambda x: [int(p) for p in x] if x.any() else []) \
        .rename('products')

    return predictions


def format_predictions(predictions):
    return predictions.apply(lambda x: ' '.join([str(p) for p in x]) if len(x) else 'None')


if __name__ == '__main__':
    orders = pd.read_csv('input/orders.csv')
    orders_product_prior = pd.read_csv('input/order_products__prior.csv')
    orders_product_train = pd.read_csv('input/order_products__train.csv')

    if eval:
        train_orders = orders[orders['eval_set'] == 'train']
        train_y = pd.merge(train_orders, orders_product_train).groupby('order_id')['product_id'] \
            .apply(lambda x: [int(p) for p in x] if x.any() else [])
        train_pred = get_predictions(train_orders)
        print('F1-score', calc_avg_f1(train_y, train_pred))
    else:
        test_orders = orders[orders['eval_set'] == 'test']
        predictions = get_predictions(test_orders)
        format_predictions(predictions) \
            .to_csv('repeat_last_reordered.csv', header=['products'], index_label=['order_id'])
