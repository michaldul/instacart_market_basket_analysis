import pandas as pd


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
    orders['days_since_first_order'] = orders.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)
    orders_prior = pd.merge(orders, prior)

    order_intervals = orders_prior\
        .groupby(['user_id', 'product_id'])['days_since_first_order']\
        .apply(product_orders_intervals_days)

    order_intervals\
        .to_csv('processed/product_orders_intervals.csv', header=['intervals'], index_label=['user_id', 'product_id'])