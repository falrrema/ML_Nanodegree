#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:23:11 2018

Exploratory Data Analysis
Project: Quora Insincere Questions Classification Project

Testing and validating preprocessing steps

@author: Fabs
"""

# Loading libraries helper functions and data ---------------------------------
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics, naive_bayes, model_selection, linear_model
from nltk.corpus import stopwords


# Loading helper functions
import helper as h
pd.set_option('float_format', '{:f}'.format)

# Loading Data
tqdm.pandas()
train_set = pd.read_csv('Data/Train_Features.csv')
print('The data has {} rows and {} columns'.format(train_set.shape[0], train_set.shape[1]))

# Defining pipeline 
def create_pipeline(learner = None, ngram = [1,1], 
                    term_count = 1, vocabulary = None):
    estimators = []
    estimators.append(('tfidf', TfidfVectorizer(ngram_range = [1,1], 
                                               min_df = term_count,
                                               vocabulary = vocabulary,
                                               lowercase = False)))    
    if (learner is not None):    
        estimators.append(('learner', learner))
        
    pipe = Pipeline(estimators)
    return(pipe)

# GridSearch with LogisticRegression() ----------------------------------------
lrn = LogisticRegression(class_weight = 'balanced', random_state = 33)
pipe_log = create_pipeline(lrn)
stop_words = h.get_clean_stopwords(stopwords.words('english'))
                   
parameters = {'tfidf__ngram_range': [(1,1), (2,2), (1,2), (2,3), (1,3)],
              'tfidf__analyzer': ['word', 'char', 'char_wb'],
              'tfidf__stop_words': [None, stop_words],
              'tfidf__max_df': [0.75, 1.0],
              'tfidf__min_df': [1, 2, 5],
              'tfidf__max_features': [None, 10000, 50000]}

grid_search = GridSearchCV(estimator=pipe_log, param_grid=parameters, cv=3, n_jobs=8, 
                           verbose=2,scoring = ['f1', 'roc_auc'], refit=False,
                           return_train_score=True)

grid_search.fit(train_set['qt_clean'].astype('U'), train_set['target'])
grid_search.transform(train_set['qt_clean'].astype('U'))

# Saving Gridsearch 
with open('Data/GridSearch_Preprocess.pkl', 'wb') as output:
    pickle.dump(grid_search, output, pickle.HIGHEST_PROTOCOL)


df_results = pd.DataFrame(grid_search.cv_results_)
df_results.to_csv('Data/Results_GS_preprocess.csv')





















