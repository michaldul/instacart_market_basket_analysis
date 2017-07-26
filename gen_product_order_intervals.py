import pandas as pd
import numpy as np

def product_orders_intervals_days(since_first_ordr_d):
    """
    >>> product_orders_intervals_days(pd.Series([0]))
    []
    >>> product_orders_intervals_days(pd.Series([11, 20, 100]))
    [9, 80]
    """
    return [since_first_ordr_d[since_first_ordr_d.index[i]] - since_first_ordr_d[since_first_ordr_d.index[i - 1]]
            for i in range(1, len(since_first_ordr_d))]


if __name__ == '__main__':
    orders = pd.read_csv('input/orders.csv')
    prior = pd.read_csv('input/order_products__prior.csv')
    orders = orders[orders['eval_set'] == 'prior']
    orders['days_since_first_order'] = orders.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)
    orders_prior = pd.merge(orders, prior)

    days_since_first_order_groupped = orders_prior.groupby(['user_id', 'product_id'])['days_since_first_order']
    order_intervals = days_since_first_order_groupped.apply(product_orders_intervals_days)
    last_order_day = days_since_first_order_groupped.max()

    df = pd.DataFrame({'intervals': order_intervals, 'product_prior_order_day': last_order_day})
    df['avg_interval'] = order_intervals.apply(np.mean)
    df['interval_std'] = order_intervals.apply(np.std)
    df['n_intervals'] = order_intervals.apply(len)

    df.to_csv('processed/product_orders_intervals.csv')