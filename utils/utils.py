from datetime import datetime, timedelta
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np 

def get_path(filename: str = 'train.csv') -> tuple[bool, str]:
    curr_path = str(os.getcwd())
    path_data = curr_path + '/data/' + filename
    file_extension = path_data.split(".")[-1]
    return path_data, file_extension

def get_dataframe(path_data: str, file_extension: str) -> pd.DataFrame:
    if file_extension == 'xlsx':
        df = pd.read_excel(path_data, engine='openpyxl')
    elif file_extension == 'xls':
        df = pd.read_excel(path_data)
    elif file_extension == 'csv':
        df = pd.read_csv(path_data)
    return df 

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

def plot_features(df):
    #substituting unscaled feature values by scaled values using sklearn StandardScaler
    scaler = StandardScaler()
    df.iloc[:,:] = scaler.fit_transform(df.iloc[:,:])

    #drawing figure with title and single axis. Size and resolution are specified
    plt.figure(figsize=(18,6),dpi=600);
    plt.title('Comparison of scaled features',fontsize=22);

    #setting y axis label
    plt.ylabel('Scaled values');

    #rotating x axis ticks by 90 degrees
    plt.xticks(rotation=90);

    #drawing boxplot of scaled feature values
    sns.boxplot(data=df);