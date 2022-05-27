

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from translate import Translator

import numpy as np
import re

def my_replace(input):
    return input.group()[1:]

def clean_tweet(row, location):
    tweet = row[location]
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

def sentiment_vader(row, location):
    text = row[location]
    translator= Translator(from_lang="spanish",to_lang="english")
    try:
        sentence = translator.translate(text)
    except: 
        print(f'Invalid translation: {text}')
        return 0

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict['compound'] #neg, neu, pos

def apply_vader(df, name_col = 0):
    df['clean'] = df.apply(lambda row: clean_tweet(row, name_col),axis=1)
    df['sentiment'] = df.apply(lambda row: sentiment_vader(row, 'clean'),axis=1)
    return df

if __name__ == "__main__":
    df = pd.DataFrame(['hola', 'bueno', 'malo', 'me encanta', "@master en https://magic.com #marvella", 'jajajaja', 'enfin', 'hooolaa'], columns=['tweet'])
    print(apply_vader(df))
