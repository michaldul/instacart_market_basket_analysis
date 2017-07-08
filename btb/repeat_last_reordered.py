
import pandas as pd

orders = pd.read_csv('../input/orders.csv')
orders_product_prior = pd.read_csv('../input/order_products__prior.csv')


def get_test_orders():
    test_orders = orders[orders['eval_set'] == 'test']
    return test_orders


def get_predictions():
    test_orders = get_test_orders()
    test_orders['prev_order'] = test_orders['order_number'] - 1
    last_orders = pd.merge(orders, test_orders,
                           left_on=['user_id', 'order_number'],
                           right_on=['user_id', 'prev_order'],
                           suffixes=('', '_test'),
                           how='inner')
    last_orders[['order_id', 'order_id_test']]
    prior_reordered = orders_product_prior[orders_product_prior['reordered'] == 1]

    return pd.merge(last_orders, prior_reordered, on=['order_id'], how='left')\
        .groupby('order_id_test')['product_id']\
        .apply(lambda x: ' '.join([str(int(p)) for p in x]) if x.any() else 'None')

get_predictions().to_csv('repeat_last_reordered.csv', header=['products'], index_label=['order_id'])