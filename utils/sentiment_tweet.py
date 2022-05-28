
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dateutil.parser import parse
from googletrans import Translator, constants
from sklearn.preprocessing import StandardScaler
import numpy as np
import re

from utils import *

def change_data(row, location):
    try:
        return parse(row[location]).strftime("%Y-%m-%d")
    except:
        return '0'

def my_replace(input):
    return input.group()[1:]

def sentiment_vader(text):
    sentence = ''
    translator = Translator()
    sentence = translator.translate(text).text
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound'] 

def clean_tweet(tweet):
    if type(tweet) == float:#np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","user", temp)
    temp = re.sub("#[A-Za-z0-9_]+", my_replace, temp)
    temp = re.sub(r'http\S+', 'url', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    return temp

def twitter_sentiment_csv(df_tweets):
    df_test = df_tweets.copy()
    df_test['clean'] = np.vectorize(clean_tweet)(df_test['text'])
    df_test['score'] = np.vectorize(sentiment_vader)(df_test['clean'])
    save_df_local(df=df_test, output_name='twitter_processed.csv', create_folder=True, new_folder_path= 'data')
    return df_test

if __name__ == '__main__':
    pass 
    #twitter_sentiment_csv(df)