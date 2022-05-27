from datetime import datetime, timedelta

from utils import *

import pandas as pd
from pytrends.request import TrendReq

def historical_pytrends(start = (datetime.now() - timedelta(days=265)).strftime("%Y-%m-%d"), 
                        end = datetime.now().strftime("%Y-%m-%d"), keys = ["Ibex 35"], cat = "0", geo="", gprop=""):
    df = pd.DataFrame()
    pytrends = TrendReq(hl='en-US')
    pytrends.build_payload(keys,
                        cat,
                        start+" "+end,
                        geo,
                        gprop)
    
    data = pytrends.interest_over_time() #--> 265 maxim!!!
    data.drop('isPartial', axis=1, inplace=True)

    return data#data.iloc[1:] --> treu 1a columna que no hi ha el % de millora

def get_all_pytrends(start = (datetime.now() - timedelta(days=265)).strftime("%Y-%m-%d"), 
                        end = datetime.now().strftime("%Y-%m-%d"), total_days=2000):
    keys = [["Ibex 35"],
          ["ibex 35 down"],
          ["credito"],
          ["recesión"]]

    for i, group in enumerate(keys):
      l = []
      for index in range(total_days//265):
        start = (datetime.now() - timedelta(days=265*(index+1))).strftime("%Y-%m-%d")
        end = (datetime.now() - timedelta(days=265*index)).strftime("%Y-%m-%d")
        l.append(historical_pytrends(start, end, keys=group))
      if i == 0:
        df = pd.concat(l)
      else:
        df = df.join(pd.concat(l))

    return df_add_improve_cols(df)


if __name__ == "__main__":
    df_pytrends = get_all_pytrends()
    save_df_local(df=df_pytrends, output_name='pytrends_data.csv', create_folder=True, new_folder_path= '../data')
    print(df_pytrends)