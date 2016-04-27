from sklearn import metrics
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
import json
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

class Tokenizer():

    def __init__(self):
        self.stemmer = LancasterStemmer()

    def __call__(self, text):
        return [self.stemmer.stem(token) for token in word_tokenize(text)]


def parse(tweet):
    try:
        return json.loads(tweet)
    except Exception:
        return {}
        
def english(tweet):
    return 'lang' in tweet and tweet['lang'] == 'en'
        
def valid(tweet):
    return tweet is not None and 'text' in tweet
    
def get_text(tweet):
    text = tweet['text']
    text = re.sub(r'https?:\/\/[^\t ]*', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    return text