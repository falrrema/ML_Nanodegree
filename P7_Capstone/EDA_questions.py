# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis
Project: Quora Insincere Questions Classification Project

The following notebook will explore the data set provided in the quora kaggle competition. 

References:
https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/
https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
https://medium.com/@kangeugine/pipeline-80a54121032
https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
http://danshiebler.com/2016-09-14-parallel-progress-bar/
"""

# Loading libraries
from sklearn import model_selection, metrics, linear_model, naive_bayes
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost
import pickle
import matplotlib.pyplot as plt

%matplotlib inline

# Loading helper functions
import helper as h
pd.set_option('float_format', '{:f}'.format)

# 1. Basic Statistics ---------------------------------------------------------
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

# 2. Text Cleaning and preprocessing  -----------------------------------------
# Applying lowercasing + punctuation removal + special character removal + 
# stopwords removal and lemmatization
print('Applying clean text processing step: '
    'lowercasing + punctuation removal + special character removal + '
    'lemmatizacion and stopword removal (separate column))')
stop_words = h.get_clean_stopwords(stopwords.words('english'))
train_set['qt_clean'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t))
train_set['qt_clean_stop'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t, stop_words = stop_words))

# 3. Basic Statistics ----------------------------------------------------------
# 3.1 Word level Statistics
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

# 3.2 Document level Statistics
# Creating several document derived features
train_set['char_count'] = train_set.qt_clean_stop.progress_apply(len) # char clean count
train_set['word_count'] = train_set.qt_clean_stop.progress_apply(lambda x: len(x.split())) # word clean count
train_set['word_density'] = train_set.char_count / (train_set.word_count+1) # word density count

# There are 66 documents with zero characters and 5723 with 1 non-stopword word
col = ['char_count', 'word_count', 'word_density']
train_set[col].describe().applymap(np.int64)
train_set.word_count.value_counts(sort=False)

train_set[train_set.word_count == 0][['question_text', 'target']]
train_set[train_set.word_count == 1][['question_text', 'target']].head(50)
train_set[(train_set.word_count == 1) & (train_set.target == 1)][['question_text', 'target']].head(50)

# And documents with more that 14 words (95% percentile)
np.percentile(train_set.word_count, [0, 25, 50, 75, 99])
sns.distplot(train_set.char_count, kde=False, axlabel = 'Character Count')
sns.distplot(train_set.word_count, kde=False, axlabel = 'Word Count')

# Target comparison
train_set[['char_count', 'target']].groupby('target').aggregate([min, np.mean, np.median, max])
train_set[['word_count', 'target']].groupby('target').aggregate([min, np.mean, np.median, max])
train_set[['word_density', 'target']].groupby('target').aggregate([min, np.mean, np.median, max])

ax = sns.boxplot(x="target", y="word_count", data=train_set[['word_count', 'target']])
ax.set(ylim = [0,40], ylabel = 'Word Count', xlabel = 'Target')
ax.set_title('Word Count by Target boxplot')

ax = sns.boxplot(x="target", y="char_count", data=train_set[['char_count', 'target']])
ax.set(ylim = [0,300], ylabel = 'Character Count', xlabel = 'Target')
ax.set_title('Character Count by Target boxplot')

ax = sns.boxplot(x="target", y="word_density", data=train_set[['word_density', 'target']])
ax.set(ylim = [0,50], ylabel = 'Word Density', xlabel = 'Target')
ax.set_title('Word Density by Target boxplot')

# Filtering Outlier Documents 
train_set = train_set[(train_set.word_count > 2) & (train_set.word_count < 15)]

# Saving Preprocessing 
with open('Data/train_preproc.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)

# 4 Document Meta features ----------------------------------------------------
# These festures may aid text feature modelling

# Basic features
train_set['char_count'] = train_set.qt_clean_stop.progress_apply(len) # char clean count
train_set['word_count'] = train_set.qt_clean_stop.progress_apply(lambda x: len(x.split())) # word clean count
train_set['word_density'] = train_set.char_count / (train_set.word_count+1) # word density count
train_set['n_stopwords'] = train_set.qt_clean.progress_apply(lambda x: len([x for x in x.split() if x in stop_words]))
train_set['n_numbers'] = train_set.qt_clean_stop.progress_apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train_set['n_upper'] = train_set.qt_clean.progress_apply(lambda x: len([x for x in x.split() if x.isupper()]))

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

col = ['char_count', 'word_count', 'word_density', 'n_stopwords', 
       'n_numbers', 'n_upper', 'polarity', 'subjectivity', 'positivity',
       'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']

# ¿Are they informative of the target? (information gain analysis)
ig_score = mutual_info_classif(train_set[col], train_set.target)
ig_score_df = pd.DataFrame({'metafeatures':col, 
                       'inf_gain':ig_score})
ig_score_df.sort_values('inf_gain', ascending = False)

# Saving Preprocessing and meta-feature creation
with open('Data/train_meta_features.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)
    
# 5. Benchmark models ---------------------------------------------------------
with open('Data/train_meta_features.pkl', 'rb') as input:
    train_set = pickle.load(input)
    
# Train Valid split
# For further analysis and to avoid test leakage I will create a  
# train, test and valid set
pretrain_x, test_x, pretrain_y, test_y = model_selection.train_test_split(train_set, 
                                                                      train_set.target,
                                                                      test_size = 0.3,
                                                                      stratify = train_set.target,
                                                                      random_state = 33)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(pretrain_x, 
                                                                      pretrain_y,
                                                                      test_size = 0.2,
                                                                      stratify = pretrain_y,
                                                                      random_state = 33)

text = train_x.qt_clean_stop


# Defining pipeline and evaluation functions
def create_simple_pipeline(learner, analyzer = 'word', ngram = [1,1], term_count = 1):
    estimators = []
    estimators.append(('vect', CountVectorizer(ngram_range = [1,1], 
                                               min_df = term_count,
                                               analyzer = 'word')))
    estimators.append(('tfidf', TfidfTransformer()))
    estimators.append(('learner', learner))
    model = Pipeline(estimators)
    return(model)

def evaluate_pipeline(pipe, X, y, cv = 5, cpus = 5, verbose = True, seed = 33):  
    kfold = model_selection.KFold(n_splits=cv, random_state=seed)
    results = model_selection.cross_validate(pipe, X, y, scoring = 'f1',
                                              cv=kfold, n_jobs=cpus, 
                                              verbose = verbose, 
                                              return_train_score=True)
    results = pd.DataFrame(results).mean(axis=0)
    return(results)

# Bechmark Models 
results = {}

# Naive Model
lrn = DummyClassifier(random_state = 33)
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, X = text, y = train_y)
results['naive'] = results_mod
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

# baseline score: Logistic Regression 
lrn_basic = linear_model.LogisticRegression(class_weight = 'balanced', random_state = 33)
pipe_lrn = create_simple_pipeline(lrn_basic)
results_mod = evaluate_pipeline(pipe_lrn, text, train_y)
results['logreg_basic'] = results_mod
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Logistic Classifer)')    
    
# 5. Feature Engineering ------------------------------------------------------
# 5.1 Feature Engineering: Ngram and noise reduction 
# One problem is the vast amount of text features generated in CountVectorizer
# much of them are pure noise
count_vect = CountVectorizer(binary = True)
count_vect.fit(text)
xtrain_count = count_vect.transform(text)
xtrain_count.shape
print('The document-term matrix of xtrain has {} terms'.format(xtrain_count.shape[1]))

# Analyzing sparsity of each column in sparse matrix
colsums = np.asarray(xtrain_count.sum(axis=0)).flatten()
colnames = count_vect.get_feature_names()
sparse_df = pd.DataFrame(np.column_stack((colnames,colsums)), columns = ['word', 'doc_count'])
sparse_df['doc_count'] = sparse_df['doc_count'].astype(int)
sparse_df['sparsity'] = (xtrain_count.shape[0] - sparse_df.doc_count)/xtrain_count.shape[0]*100
sparse_df = sparse_df.sort_values(['doc_count'], ascending = False)

above_99 = sparse_df[sparse_df.sparsity >= 99].shape[0]
above_95 = sparse_df[sparse_df.sparsity >= 95].shape[0]
above_90 = sparse_df[sparse_df.sparsity >= 90].shape[0]
one_term = sparse_df[sparse_df.doc_count ==1].shape[0]
two_term = sparse_df[sparse_df.doc_count <=2].shape[0]
five_term = sparse_df[sparse_df.doc_count <=5].shape[0]
print('A total of {} ({}% total) terms have a sparsity above 0.99'.format(above_99, round(above_99/xtrain_count.shape[1]*100,2)))
print('A total of {} ({}% total) terms have a sparsity above 0.90'.format(above_90, round(above_90/xtrain_count.shape[1]*100,2)))
print('A total of {} ({}% total) terms appear in only 1 document'.format(one_term, round(one_term/xtrain_count.shape[1]*100,2)))
print('A total of {} ({}% total) terms appear at least in 2 document'.format(two_term, round(two_term/xtrain_count.shape[1]*100,2)))
print('A total of {} ({}% total) terms appear at least in 5 document'.format(five_term, round(five_term/xtrain_count.shape[1]*100,2)))

# Filter noise words (term count < 3) check performance with baseline
# Logistic Regression without noise terms
pipe_lrn = create_simple_pipeline(lrn_basic, term_count = 3)
results_mod = evaluate_pipeline(pipe_lrn, text, train_y) 

# The elimination of ~70% of terms showed same performance to baseline CV
# And reduced train to test CV scores thus reducing overfit
print('The mean train {} and test CV {} of noise reduce data.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Logistic Classifer)')    

# ¿What if we use bigrams and trigrams? Our dimensionality increases significantly
count_vect = CountVectorizer(binary = True, ngram_range = [1,3])
count_vect.fit(text)
xtrain_count = count_vect.transform(text)
xtrain_count.shape
print('The DTM of bi-trigrams has {}MM terms'.format(round(xtrain_count.shape[1]/1e6,1)))

# Logistic Regression with bigram and trigrams ngrams
# Shows same performance to base test CV, but better in train CV
pipe_lrn = create_simple_pipeline(lrn_basic, ngram =[1,3], term_count = 1)
results_mod = evaluate_pipeline(pipe_lrn, text, train_y)

# Performance improved by using bigram and trigrams
print('The mean train {} and test CV {} of ngrams.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

# Following the same criteria as before by filtering noise terms 
count_vect = CountVectorizer(binary = True, ngram_range = [1,3], min_df = 2)
count_vect.fit(text)
xtrain_count = count_vect.transform(text)
xtrain_count.shape
print('The DTM of bi-trigrams has reduce to {} terms'.format(xtrain_count.shape[1]))

pipe_lrn = create_simple_pipeline(lrn_basic, ngram = [1,3], term_count = 3)
results_mod = evaluate_pipeline(pipe_lrn, text, train_y)

# Logistic Regression performance decreases by filtering noise words when using bi-trigrams
# but is the same comparing to baseline. Noise reduction reduces overfitting. 
print('The mean train {} and test CV {} of ngrams.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Logistic Classifer)')    

# 5.2 Feature Engineering: Topic Modelling LSA ---------------------------
# Another way to reduce dimension and noise is to perform Latent Semantic Analysis
# LSA uses truncated SVD which reduces dimensionality of the DTM matrix
# With the resulting matrices LSA can then construct topics which may be use as a feature

# Get Bigrams
collocations = h.get_collocations(text)
bigramer = Phraser(collocations['bigramer']) # faster implementation
train_tokens = [sentence.split() for sentence in text]
text_bigram = [' '.join(bigramer[tokens]) for tokens in tqdm(train_tokens)] # replace back

# Get Trigrams
collocations = h.get_collocations(text_bigram)
trigramer = Phraser(collocations['bigramer']) # faster implementation
train_tokens = [sentence.split() for sentence in text_bigram]
bi_tri_tokens = [trigramer[tokens] for tokens in tqdm(train_tokens)] 

model_list, coherence_values = h.compute_coherence_values(bi_tri_tokens, # Determine the number of topics 
                                                          start = 2, 
                                                          stop = 30, 
                                                          step = 2) 

x = pd.DataFrame({'Topic': range(2, 30, 2), 'Coherence': coherence_values})
ax = sns.pointplot(x = 'Topic', y = 'Coherence', data = x, linestyles=["--"]) # 2 topics

# Saving Topic Results
with open('Data/Topic_LSA_Results.pkl', 'wb') as output:
    pickle.dump({'model_lis': model_list, 
                 'coherence_values': coherence_values, 
                 'topic_df': x}, output, pickle.HIGHEST_PROTOCOL)

# Inspecting 20 topics
[x.num_topics == 2 for x in model_list]
lsa_model = model_list[0]
lsa_model.print_topics(num_topics=2, num_words=20)

# Asigning topic to train_set
# Get Bigrams
train_tokens = [sentence.split() for sentence in train_set.qt_clean_stop]
text_bigram = [' '.join(bigramer[tokens]) for tokens in tqdm(train_tokens)] # replace back

# Get Trigrams
train_tokens = [sentence.split() for sentence in text_bigram]
bi_tri_tokens = [trigramer[tokens] for tokens in tqdm(train_tokens)] 

dictionary,doc_term_matrix = h.prepare_corpus(bi_tri_tokens, min_doc = 3)
test = [h.get_topics(lsa_model, x) for x in tqdm(doc_term_matrix)]

# Saving topic feature
with open('Data/train_lsa_feat.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)
    
# 5.3 Feature Engineering: Feature Selection ----------------------------------
# Another posibility is to performe feature selection on the text vectors
# Chi-Squared for Feature Selection
tfidf = TfidfVectorizer(ngram_range = [1,3], analyzer='word')  
train_tfidf = tfidf.fit_transform(text) 
chi2score = chi2(train_tfidf, train_y)
scores_df = pd.DataFrame({'words':tfidf.get_feature_names(), 
                       'chi_squared':chi2score[0],
                       'prob':chi2score[1]})

# Get features that have a prob << 0.001
best_chi = scores_df[scores_df.prob <= 0.001].sort_values(['chi_squared'], ascending = False)

N = 50
best_N_chi = best_chi[:N].sort_values(['chi_squared'])

# Most informative ngrams
plt.figure(figsize=(10,10))
plt.barh(range(N),best_N_chi.chi_squared, align='center', alpha=0.5)
plt.yticks(range(N), best_N_chi.words)
plt.xlabel('Chi_squared')

# Construct new vocabulary with these words and run logistic reg
tfidf = TfidfVectorizer(ngram_range = [1,3], vocabulary = best_chi.words)  
train_tfidf = tfidf.fit_transform(text) 

# Logistic Regression with feature selection
# Feature selection allows to reduce barely afecting baseline performance
results_mod = evaluate_pipeline(lrn_basic, train_tfidf, train_y)
print('The mean train {} and test CV {} of ngrams.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

# 6. Modelling ----------------------------------------------------------------

models = {'logreg_basic': linear_model.LogisticRegression(class_weight = 'balanced', random_state = 33),
          'ridge': 
          'naive_bayes': naive_bayes.ComplementNB(random_state = 33),
          'extratrees': ExtraTreesClassifier(random_state = 33,class_weight = 'balanced', warm_start=True),
          'adaboost': AdaBoostClassifier(random_state = 33)}


def testing_models    
    
    
lrn = linear_model.ElasticNet(random_state = 33)
pipe_lrn = create_simple_pipeline(lrn)
results_mod = evaluate_pipeline(pipe_lrn, text, train_y)
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (NB Classifer)')


h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Ada Classifer)')




parameters = {'n_estimators' : [50,100,200,400], 'learning_rate' : [0.001,0.01,0.1,1]}





