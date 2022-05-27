import sys
import os 
import inspect
from datetime import datetime, timedelta


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

#from scraper.utils import *

import pandas as pd
import numpy as np 

def save_df_local(df: pd.DataFrame, output_name: str='results.csv', create_folder: bool=False, new_folder_path:str = '../../data/scraper', compressed: bool=False ):
    if create_folder:
        os.makedirs(new_folder_path, exist_ok=True) #'folder/subfolder'
        output_name = new_folder_path + '/' + output_name

    if compressed:
        name_compressed = output_name.split(".")[0] + '.zip'
        compression_opts = dict(method='zip',archive_name=output_name.split("/")[-1])  
        df.to_csv(name_compressed, index=False,compression=compression_opts) 
    else:
        df.to_csv(output_name,index=True) 
    return


def classificator(row, location):
    if row[location] >= 1.25:
      return 4
    elif row[location] >= 0.35:
      return 3
    elif row[location] <= -1.25:
      return 0
    elif row[location] <= -0.35:
      return 1
    else:
      return 2

def df_add_improve_cols(df):
    for col in df:
          dfx = df[col]
          df[col+"%"] = np.asarray([(((dfx[x]+1e-3) / (dfx[x-1]+1e-3)) -1) * 100 if x>0 else 0 for x in range(len(dfx))]).round(2)
    return df.iloc[1:] #we cannot calculate improvement on 1st row

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')