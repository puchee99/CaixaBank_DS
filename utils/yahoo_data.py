from datetime import datetime, timedelta
import itertools

from utils import *

from yahoofinancials import YahooFinancials
import numpy as np
import pandas as pd


def historical_yahoo_one(start = (datetime.now() - timedelta(days=20000)).strftime("%Y-%m-%d"),
                    end= datetime.now().strftime("%Y-%m-%d"), ticker = '^IBEX', rename=True, featuring=True):#,"TSLA",'WFC', 'BAC', 'C' ,'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'PRLAX', 'QASGX', 'HISFX']):
    yahoo_financials = YahooFinancials(ticker)
    historical_stock_prices = yahoo_financials.get_historical_price_data(start, end, 'daily')
    data = pd.DataFrame(historical_stock_prices[ticker]['prices'])
    df = pd.DataFrame()
    df['date'] = data['formatted_date']
    df['open'] = np.nan_to_num(np.asarray(data['open']))
    df['close'] = np.nan_to_num(np.asarray(data['close']))
    df['%'] = ((df.close / df.open) -1) * 100
    df['volume'] = np.asarray(data['volume'])
    df['Y'] = df.apply(lambda row: classificator(row, "%"),axis=1)
    df['target'] = np.append(df.apply(lambda row: classificator(row, "%"),axis=1)[1:],[None])
    return df

if __name__ == "__main__":
    df_test = historical_yahoo_one()
    print(df_test)
    save_df_local(df=df_test, output_name='test_yahoo.csv', create_folder=True, new_folder_path= '../data')
