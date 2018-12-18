# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis
Project: Quora Insincere Questions Classification Project

The following notebook will explore the data set provided in the quora kaggle competition. 
"""

# Loading libraries
from sklearn import model_selection, metrics, linear_model, naive_bayes
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from wordcloud import WordCloud
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from collections import Counter 
import pandas as pd
import seaborn as sns
import xgboost
import pickle

%matplotlib inline

# Loading helper functions
import helper as h

# 1. Basic Statistics
# Loading Data
tqdm.pandas()
train_set = pd.read_csv('Data/train.csv', encoding = 'latin1')
train_set.sample(5, random_state = 3)
print('The data has {} rows and {} columns'.format(train_set.shape[0], train_set.shape[1]))

# Target variable count
ax = sns.countplot(y="target", data=train_set, palette="Set1") # strong class imbalance 
cnts = train_set.target.value_counts()
print("Strong class imbalance. " 
      "There are {} of insincere question that represent"
      "the {}% of the data".format(cnts[1], round(cnts[1]/sum(cnts)*100, 1)))

# 2. Text Cleaning and preprocessing 
# Applying lowercasing + punctuation removal + special character removal + 
# stopwords removal and lemmatization
print('Applying clean text processing step: '
    'lowercasing + punctuation removal + special character removal + '
    'lemmatizacion and stopword removal (separate column))')
stop_words = h.get_clean_stopwords(stopwords.words('english'))
train_set['qt_clean'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t))
train_set['qt_clean_stop'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t, stop_words = stop_words))

# 3. Word Statistics
# WordCloud of unigram sincere questions
sincere_questions = train_set[train_set.target == 0].qt_clean
insincere_questions = train_set[train_set.target == 1].qt_clean

h.plot_wordcloud(sincere_questions, max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2)

# WordCloud of unigram insincere questions
h.plot_wordcloud(insincere_questions, max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2)

# Word Frequencies by target
wf_sincere = h.word_frequency(train_set.qt_clean_stop[train_set.target==0])
wf_insincere = h.word_frequency(train_set.qt_clean_stop[train_set.target==1])

wf_sincere.head(10)
wf_insincere.head(10)
h.comparison_plot(wf_sincere[:20],wf_insincere[:20],'word','wordcount', .35)

# Bigram Frequencies by target
wf_sincere = h.ngram_frequency(train_set.qt_clean_stop[train_set.target==0], 2)
wf_insincere = h.ngram_frequency(train_set.qt_clean_stop[train_set.target==1], 2)

wf_sincere.head(10)
h.plot_wordcloud(' '.join(wf_sincere.ngram), max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2, collocations = False)

wf_insincere.head(10)
h.plot_wordcloud(' '.join(wf_insincere.ngram), max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2, collocations = False)

h.comparison_plot(wf_sincere[:20],wf_insincere[:20],'ngram','count', .7)

# Saving Preprocessing
with open('Data/train_preproc.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)
    
# 4. Benchmark models 
with open('Data/train_preproc.pkl', 'rb') as input:
    train_set = pickle.load(input)
    
# Train Valid split
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_set.qt_clean, 
                                                                      train_set.target,
                                                                      test_size = 0.3,
                                                                      stratify = train_set.target,
                                                                      random_state = 33)

# Defining pipeline and evaluation functions
def create_simple_pipeline(learner, ngram = 1):
    estimators = []
    estimators.append(('vect', CountVectorizer(ngram_range = [1,ngram])))
    estimators.append(('tfidf', TfidfTransformer()))
    estimators.append(('learner', learner))
    model = Pipeline(estimators)
    return(model)

def evaluate_pipeline(pipe, X, y, cv = 5, cpus = 5, verbose = True, seed = 33):  
    kfold = model_selection.KFold(n_splits=cv, random_state=seed)
    results = model_selection.cross_val_score(pipe, X, y, scoring = 'f1',
                                              cv=kfold, n_jobs=cpus, verbose = verbose)
    return(results)

# Bechmark Models
results = {}

# Naive Model
lrn = DummyClassifier(random_state = 33)
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, X = train_x, y = train_y)
results['naive'] = results_mod
print('The mean crossvalidated F1-Score was {}.'.format(round(results_mod.mean(), 2)))

# baseline score: Logistic Regression 
lrn = linear_model.LogisticRegression()
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, train_x, train_y)
results['logreg_basic'] = results_mod
print('The mean crossvalidated F1-Score was {}.'.format(round(results_mod.mean(), 2)))

h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Logistic Classifer)')

# Naive Bayes
lrn = naive_bayes.ComplementNB()
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, train_x, train_y)
results['naive_bayes'] = results_mod
print('The mean crossvalidated F1-Score was {}.'.format(round(results_mod.mean(), 2)))

h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (NB Classifer)')

# Logistic Bag Model
lrn = BaggingClassifier(linear_model.LogisticRegression(), n_estimators=10,
                        max_samples=0.5, max_features=1)
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, train_x, train_y)
results['logregbag'] = results_mod
print('The mean crossvalidated F1-Score was {}.'.format(round(results_mod.mean(), 2)))

h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (NB Classifer)')

# Xgboost
lrn = xgboost.XGBClassifier(max_depth=10, learning_rate=0.05, subsample=0.8,
                            n_estimators=200, objective='binary:logistic',
                            colsample_bytree=0.7, eta=0.3, gamma=5,
                            random_state=33)
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, train_x, train_y, cpus = 3)
results['xgboost'] = results_mod
print('The mean crossvalidated F1-Score was {}.'.format(round(results_mod.mean(), 2)))

h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (XG Classifer)')

# 5. Feature Engineering: Dimensionality Reduction
# One problem is the vast amount of text features generated in CountVectorizer
# much of them are pure noise, which 



























