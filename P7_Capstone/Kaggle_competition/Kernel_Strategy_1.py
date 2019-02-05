#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:18:27 2018

Project: Quora Insincere Questions Classification Project

First submission Kernel. The result of udacity capstone.

@author: Fabs
"""

# 1. Loading libraries helper functions and data ---------------------------------
import pandas as pd
import numpy as np
import pickle
import spacy
import string as st
import re
import lightgbm as lgb
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics, naive_bayes, model_selection
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve, precision_recall_curve
from nltk.corpus import stopwords
from textblob import TextBlob, Blobber, Word
from textblob.sentiments import NaiveBayesAnalyzer
from collections import Counter

nlp = spacy.load('en', disable=['parser', 'tagger'])

# Loading helper functions
def clean_text(text, lowering = True, remove_punct = True,
               lemmatization = True, special_char = True, stop_words = None):
    
    # Word tokenization
    words = text.split()

    # Lowercasing
    if (lowering):
        words = [word.lower() for word in words]
        
    # Special Character removal
    if (special_char):
        words = [re.sub(r'[^\x00-\x7F]+','', word) for word in words]

    # Punctuation marks removal
    if (remove_punct):
        translator = str.maketrans('', '', st.punctuation)
        # words = filter(lambda x: x != "'s", words) # special case
        words = [word.translate(translator) for word in words]
        words = filter(None, words)
        words = [word for word in words]
    
    # Stopwords removal
    if (stop_words is not None):
        words = [word for word in words if word not in stop_words] 
    
    # Lemmatization
    if (lemmatization): 
        words = [Word(word).lemmatize(pos='v') for word in words]  
    
    # Words to sentence
    sentence = " ".join(words)
    return sentence

def basic_features(df):
    stop_words = [clean_text(stop) for stop in stopwords.words('english')]
    df['char_count'] = df['qt_clean'].progress_apply(len) # char clean count
    df['word_count'] = df['qt_clean'].progress_apply(lambda x: len(x.split())) # word clean count
    df['word_density'] = df.char_count / (train_set.word_count+1) # word density count
    df['n_stopwords'] = df['qt_clean'].progress_apply(lambda x: len([x for x in x.split() if x in stop_words]))
    df['n_numbers'] = df['qt_clean'].progress_apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    df['n_upper'] = df['question_text'].progress_apply(lambda x: len([x for x in x.split() if x.isupper()]))
    df['upper_density'] = df.n_upper / (train_set.word_count+1) # upper case word density count
    df['punct_count'] = df['question_text'].progress_apply(lambda x: len("".join(_ for _ in x if _ in st.punctuation))) 
    return(df)

def sent_features(df, sentimenter):
    linguistic_feat = df['qt_clean'].progress_apply(lambda x: TextBlob(x))
    df['polarity'] = [sent.sentiment[0] for sent in tqdm(linguistic_feat)]
    df['subjectivity'] = [sent.sentiment[1] for sent in tqdm(linguistic_feat)]
    
    sentiments = [sentimenter(text) for text in tqdm(df['qt_clean'])]
    df['positivity'] = [sent.sentiment[1] for sent in tqdm(sentiments)]
    return(df)

def count_tag(text):
    counts = Counter(token[1] for token in text.tags)
    return counts
    
def check_pos_tag(tags, flag):
    pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    } 
    get_tags = [tag for tag in tags if tag in pos_family[flag]]
    get_sums = sum([tags[tag] for tag in get_tags])
    return get_sums

def postag_features(df):
    linguistic_feat = df['qt_clean'].progress_apply(lambda x: TextBlob(x))
    get_pos_tag = [count_tag(row) for row in tqdm(linguistic_feat)]
    df['noun_count'] = [check_pos_tag(tag, 'noun') for tag in tqdm(get_pos_tag)]
    df['verb_count'] = [check_pos_tag(tag, 'verb') for tag in tqdm(get_pos_tag)]
    df['adj_count'] = [check_pos_tag(tag, 'adj') for tag in tqdm(get_pos_tag)]
    df['adv_count'] = [check_pos_tag(tag, 'adv') for tag in tqdm(get_pos_tag)]
    df['pron_count'] = [check_pos_tag(tag, 'pron') for tag in tqdm(get_pos_tag)]
    return(df)

def count_ents(nlp_obj):
    counts = Counter(ent.label_ for ent in nlp_obj.ents)
    return counts

def entity_features(df, nlp_obj):
    entity_feat = []
    for doc in tqdm(nlp.pipe(df['qt_clean'], batch_size=40000, n_threads=4)):
        entity_feat.append(doc)
    
    ents = [count_ents(ent) for ent in tqdm(entity_feat)]
    df['person'] = [ent['PERSON'] for ent in tqdm(ents)]
    df['norp'] = [ent['NORP'] for ent in tqdm(ents)]
    df['fac'] = [ent['FAC'] for ent in tqdm(ents)]
    df['org'] = [ent['ORG'] for ent in tqdm(ents)]
    df['gpe'] = [ent['GPE'] for ent in tqdm(ents)]
    df['loc'] = [ent['LOC'] for ent in tqdm(ents)]
    df['prod'] = [ent['PRODUCT'] for ent in tqdm(ents)]
    df['date'] = [ent['DATE'] for ent in tqdm(ents)]
    df['time'] = [ent['TIME'] for ent in tqdm(ents)]
    df['quant'] = [ent['QUANTITY'] for ent in tqdm(ents)]
    df['ord'] = [ent['ORDINAL'] for ent in tqdm(ents)]
    df['card'] = [ent['CARDINAL'] for ent in tqdm(ents)]
    return(df)

def create_pipeline(learner = None, ngram = [1,1], max_features = None, 
                    min_term = 1, max_term = 1.0, vocabulary = None):
    estimators = []
    estimators.append(('vect', CountVectorizer(ngram_range = [1,1], 
                                               min_df = min_term,
                                               max_df = max_term,
                                               max_features = max_features,
                                               vocabulary = vocabulary)))
    estimators.append(('tfidf', TfidfTransformer()))
    
    if (learner is not None):    
        estimators.append(('learner', learner))
        
    pipe = Pipeline(estimators)
    return(pipe)

def threshold_search(y_true, y_proba):
    'https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/75735'
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result 
   
    
def testing_models(train_x, train_y, test_x, test_y, models):
    
    mod_results = {}
    
    for model in models:
        learner = models[model]
        results = {}
        print('Modelling with', model, '...')
        
        learner.fit(train_x, train_y)
        predictions_train = learner.predict_proba(train_x)
        predictions_test = learner.predict_proba(test_x)
        
        cut_off = threshold_search(train_y, predictions_train[:,1])
        pred_test_threshol = predictions_test[:,1] > cut_off['threshold']
        
        results['train_score'] = cut_off['f1']
        results['test_score'] = metrics.f1_score(test_y, pred_test_threshol)
        results['threshold'] = cut_off['threshold']
        results['fitted_model'] = learner
        
        print(results)
        mod_results[model] = results   
    return(mod_results)


# Loading Data
tqdm.pandas()
train_set = pd.read_csv('Data/train.csv')
test_set = pd.read_csv('Data/test.csv')

# 2. Text Cleaning ------------------------------------------------------------
# Stopword removal did not show an improvement
train_set['qt_clean'] = train_set['question_text'].progress_apply(lambda t: clean_text(t))
test_set['qt_clean'] = test_set['question_text'].progress_apply(lambda t: clean_text(t))

# 3. Document Meta features ---------------------------------------------------
# These features may aid text feature modelling  
# Basic features
train_set = basic_features(train_set)
test_set = basic_features(test_set)

# Sentiment features
sentimenter = Blobber(analyzer=NaiveBayesAnalyzer())
train_set = sent_features(train_set, sentimenter)
test_set = sent_features(test_set, sentimenter)

# Linguistic Features
train_set = postag_features(train_set)
test_set = postag_features(test_set)

# Entity Features
train_set = entity_features(train_set, nlp)
test_set = entity_features(test_set, nlp)

# Saving Preprocessing 
#with open('Data/train_preproc.pkl', 'wb') as output:
#    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)
#
#with open('Data/test_preproc.pkl', 'wb') as output:
#    pickle.dump(test_set, output, pickle.HIGHEST_PROTOCOL)

# 4. Top ChiSquared Text Features ---------------------------------------------
with open('Data/train_preproc.pkl', 'rb') as input:
    train_set = pickle.load(input)

with open('Data/test_preproc.pkl', 'rb') as input:
    test_set = pickle.load(input)

# Chi-Squared for Feature Selection
tfidf = TfidfVectorizer(ngram_range = [1,2])  
train_tfidf = tfidf.fit_transform(train_set.qt_clean) 
chi2score = chi2(train_tfidf, train_set['target'])
scores_df = pd.DataFrame({'words':tfidf.get_feature_names(), 
                       'chi_squared':chi2score[0],
                       'prob':chi2score[1]})

stop_words = [clean_text(stop) for stop in stopwords.words('english')]
scores_df = scores_df[~scores_df['words'].isin(stop_words)]
best_chi = scores_df.sort_values(['chi_squared'], ascending = False)[0:100]

# Construct new vocabulary with these words 
tfidf_chi = TfidfVectorizer(ngram_range = [1,2], vocabulary = best_chi['words'])  
train_tfidf = tfidf_chi.fit_transform(train_set.qt_clean) 
test_tfidf = tfidf_chi.transform(test_set.qt_clean)

dense_train = pd.DataFrame(train_tfidf.todense(), columns = tfidf_chi.get_feature_names())
dense_test = pd.DataFrame(test_tfidf.todense(), columns = tfidf_chi.get_feature_names())

train_set_words = pd.concat([train_set, dense_train], axis=1)
test_set_words = pd.concat([test_set, dense_test], axis=1)

# 5. Model Prediction Features ------------------------------------------------
train_x, valid_x, train_y, valid_y = train_test_split(train_set,
                                                    train_set['target'],
                                                    test_size = 0.3,
                                                    stratify = train_set['target'],
                                                    random_state = 33)

pipe_vect = create_pipeline(ngram = [1,2], max_term=0.75, min_term=5, 
                            max_features=50000)
train_vect = pipe_vect.fit_transform(train_x.qt_clean)
valid_vect = pipe_vect.transform(valid_x.qt_clean)
#test_vect = pipe_vect.transform(test_set.qt_clean)

# Defining Models to use
log = LogisticRegression(class_weight = 'balanced', random_state = 33)
nb = naive_bayes.ComplementNB()
lgbm = lgb.LGBMClassifier(colsample_bytree = 0.8, class_weight = 'balanced'
                        subsample = 0.8, learning_rate = 0.05, 
                        max_depth = 4, , objective = 'binary', n_jobs = 4,
                        n_estimators = 1000, random_state = 33)

models = {
          'lgbm': lgbm
          }

text_results = testing_models(train_vect, train_y, valid_vect, valid_y, models)

models = {'log': log,
          'nb': nb,
          'lgbm': lgbm
          }
