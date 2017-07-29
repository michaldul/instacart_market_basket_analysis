import pandas as pd

if __name__ == '__main__':

    none_df = pd.read_csv('none_prediction.csv')
    none_orders = none_df[none_df['none'] == 1]

    predictions_df = pd.read_csv('xgboost.csv')
    predictions_df.loc[predictions_df['order_id'].isin(none_orders['order_id']), 'products'] = 'None'

    predictions_df.to_csv('merged.csv', index=False)