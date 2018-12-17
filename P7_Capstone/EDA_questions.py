# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Loading libraries
from sklearn import model_selection, metrics, linear_model
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

%matplotlib inline

# Loading helper functions
import helper as h

# 1. Basic Statistics
# Loading Data
tqdm.pandas()
train_set = pd.read_csv('Data/train.csv', encoding = 'latin1')
train_set.sample(5)

train_set.shape

# Target variable count
ax = sns.countplot(y="target", data=train_set, palette="Set1") # strong class imbalance 
cnts = train_set.target.value_counts()
print("Strong class imbalance. " 
      "There are {} of insincere question that represent"
      "the {}% of the data".format(cnts[1], round(cnts[1]/sum(cnts)*100, 1)))

# 2. Text Cleaning and preprocessing 
# Applying lowercasing + punctuation removal + special character removal + 
# stopwords removal and lemmatization
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

# Getting Concept words by collocations
train_tokens = [t.split() for t in train_set.qt_clean_stop]
bigramer = Phrases(train_tokens)  # train model

train_set.qt_clean_stop[100]
[bigramer[t] for t in train_tokens[100:101]]

bigram_counter = list()
bigram_list = list(bigramer.vocab.items())
for key, value in bigram_list:
    str_key = key.decode()
    if len(str_key.split("_")) > 1:
        bigram_counter.append(tuple([str_key, value]))

bigram_df = pd.DataFrame(bigram_counter, columns=['bigrams', 'count'])
bigram_df.sort_values('count', ascending = False).head(10)

# Replacing collocations in questions to improve interpretability
bigramer = Phraser(bigramer) # faster implementation
train_set['qt_col_stop'] = [' '.join(bigramer[tokens]) for tokens in tqdm(train_tokens)]

wf_sincere = h.word_frequency(train_set.qt_col_stop[train_set.target==0])
wf_insincere = h.word_frequency(train_set.qt_col_stop[train_set.target==1])

wf_sincere[['_' in i for i in wf_sincere.word]].head(10)
wf_insincere[['_' in i for i in wf_insincere.word]].head(10)

train_set.to_csv('Data/train_preproc.csv', sep='\t')

# 4. Modelling
train_set = pd.read_csv('Data/train_preproc.csv', encoding = 'latin1', sep = '\t')

# Train Valid split
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_set.qt_col_stop, 
                                                                      train_set.target,
                                                                      test_size = 0.3,
                                                                      stratify = train_set.target,
                                                                      random_state = 33)

# Defining pipeline and evaluation functions
def create_simple_pipeline(learner):
    estimators = []
    estimators.append(('vect', CountVectorizer()))
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
# Naive Model
lrn = DummyClassifier(random_state = 33)
pipe_lrn = create_simple_pipeline(lrn)
results_dummy = evaluate_pipeline(pipe_lrn, X = train_x, y = train_y)
results_dummy.mean()
h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Dummy Classifer)')

# baseline score: Logistic Regression 
lrn = linear_model.LogisticRegression()
pipe_lrn = create_simple_pipeline(lrn)
results_log = evaluate_pipeline(pipe_lrn, train_x, train_y)
results_log.mean()
h.plot_learning_curve(pipe_lrn, train_x, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Dummy Classifer)')

train_set.columns
