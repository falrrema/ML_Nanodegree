#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:37:57 2018

Exploratory Data Analysis
Project: Quora Insincere Questions Classification Project

The following code will create clean text columns and extract meta features.

@author: Fabs
"""

# Loading libraries helper functions and data ---------------------------------
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer

# Loading helper functions
import helper as h
pd.set_option('float_format', '{:f}'.format)

# Loading Data
tqdm.pandas()
train_set = pd.read_csv('Data/train.csv', encoding = 'latin1')
print('The data has {} rows and {} columns'.format(train_set.shape[0], train_set.shape[1]))

# 2. Text Cleaning, preprocessing, and Meta Feature extraction ----------------
# 2.1 Applying lowercasing + punctuation removal + special character removal + 
# stopwords removal and lemmatization
print('Applying clean text processing step: '
    'lowercasing + punctuation removal + special character removal + '
    'lemmatizacion and stopword removal (separate column))')
stop_words = h.get_clean_stopwords(stopwords.words('english'))
train_set['qt_clean'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t))
train_set['qt_clean_stop'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t, stop_words = stop_words))

# 2.2 Document Meta features 
# These features may aid text feature modelling  
# Basic features
train_set['char_count'] = train_set.qt_clean_stop.progress_apply(len) # char clean count
train_set['word_count'] = train_set.qt_clean_stop.progress_apply(lambda x: len(x.split())) # word clean count
train_set['word_density'] = train_set.char_count / (train_set.word_count+1) # word density count
train_set['n_stopwords'] = train_set.qt_clean.progress_apply(lambda x: len([x for x in x.split() if x in stop_words]))
train_set['n_numbers'] = train_set.qt_clean_stop.progress_apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train_set['n_upper'] = train_set.question_text.progress_apply(lambda x: len([x for x in x.split() if x.isupper()]))
train_set['title_word_count'] = train_set.question_text.progress_apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

# Sentiment features
train_set['polarity'] = train_set.qt_clean.progress_apply(lambda x: TextBlob(x).sentiment[0])
train_set['subjectivity'] = train_set.qt_clean.progress_apply(lambda x: TextBlob(x).sentiment[1])

sentimenter = Blobber(analyzer=NaiveBayesAnalyzer())
train_set['positivity'] = train_set.qt_clean.progress_apply(lambda x: h.NB_sentimenter(x, sentimenter))

# Linguistic Features
train_set['noun_count'] = h.parallel_process(train_set.qt_clean, h.check_nouns)
train_set['verb_count'] = h.parallel_process(train_set.qt_clean, h.check_verbs)
train_set['adj_count'] = h.parallel_process(train_set.qt_clean, h.check_adj)
train_set['adv_count'] = h.parallel_process(train_set.qt_clean, h.check_adv)
train_set['pron_count'] = h.parallel_process(train_set.qt_clean, h.check_pron)

# Saving
train_set.columns
train_set = train_set[['qid', 'question_text', 'qt_clean', 'qt_clean_stop', 'char_count', 
           'word_count', 'word_density', 'n_stopwords', 'n_numbers',
           'n_upper', 'title_word_count', 'polarity', 'subjectivity', 
           'positivity', 'noun_count', 'verb_count', 'adj_count', 
           'adv_count', 'pron_count', 'target']]

train_set.to_csv('Data/Train_Features.csv', index = False)






