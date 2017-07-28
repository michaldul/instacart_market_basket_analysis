import pandas as pd


def calc_streak(orders_numbers):
    orders_numbers = orders_numbers.reset_index()
    max_order_num = orders_numbers['number_of_orders'][0]
    order_numbers = list(orders_numbers['order_number'])
    return _calc_streak(order_numbers, max_order_num)


def _calc_streak(order_numbers, max_order_num):
    """
    >>> _calc_streak([], 5)
    0
    >>> _calc_streak([1, 2, 6, 7, 8], 8)
    3
    """
    streak = 0
    idx = max_order_num
    for i in reversed(order_numbers):
        if idx == i:
            idx -= 1
            streak += 1
        else:
            break

    return streak


if __name__ == '__main__':
    orders = pd.read_csv('../input/orders.csv')
    prior = pd.read_csv('../input/order_products__prior.csv')


    orders_prior = pd.merge(orders, prior)

    orders_prior = pd.merge(orders_prior,
                            pd.DataFrame(orders_prior.groupby('user_id')['order_number'].agg('max').rename('number_of_orders')),
                            left_on='user_id',
                            right_index=True)

    # 2.5h computation
    streak = orders_prior.groupby(['user_id', 'product_id'])[['order_number', 'number_of_orders']].apply(calc_streak)
    streak = streak.rename('product_streak_last_order')

    output = pd.merge(streak.reset_index(),
                      pd.DataFrame(orders_prior.groupby('user_id')['order_number'].agg('max').rename(
                          'number_of_client_orders')).reset_index())

    output.to_csv('../processed/product_streak.csv', index=False)