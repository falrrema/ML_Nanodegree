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
from sklearn import metrics, naive_bayes, model_selection, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser
from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer

import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm
import pickle
import matplotlib.pyplot as plt


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

# Creating several document derived features
train_set['char_count_full'] = train_set.question_text.progress_apply(len) # char clean count
train_set['word_count_full'] = train_set.question_text.progress_apply(lambda x: len(x.split())) # word clean count
train_set['word_density_full'] = train_set.char_count_full / (train_set.word_count_full+1) # word density count

col = ['char_count_full', 'word_count_full', 'word_density_full']
pd.DataFrame(train_set.word_count_full.value_counts(sort=False).head(10))

# Summary statistics
train_set[col].describe().applymap(np.int64)

train_set[train_set.word_density_full >= 43].question_text
train_set.question_text[455194].split()
sns.distplot(train_set.char_count_full, kde=False, axlabel = 'Character Count')

iqr = np.subtract(*np.percentile(train_set.char_count_full, [75, 25])) #interquartile range
upper_bound_outlier = np.percentile(train_set.char_count_full, 75) + 1.5*iqr
print('Above {} characters documents are considered outliers'.format(upper_bound_outlier))


iqr = np.subtract(*np.percentile(train_set.word_density_full, [75, 25])) #interquartile range
upper_bound_outlier = np.percentile(train_set.word_density_full, 75) + 1.5*iqr
print('Above {} word density documents are considered outliers'.format(upper_bound_outlier))

train_set[['word_count_full', 'target']].groupby('target').aggregate([min, np.mean, np.median, max])

criteria = (train_set.word_count_full < 3) | (train_set.char_count_full > 200) | (train_set.word_density_full > 10)
outliers = train_set[criteria]
proportion = 4458/(4458+12337)*100
print('Beign an outlier makes the insincere class proportion significantly higher ({}%)'.format(round(proportion)))
outliers['target'].value_counts()



# 2. Text Cleaning, preprocessing, and Meta Feature extraction ----------------
# 2.1 Applying lowercasing + punctuation removal + special character removal + 
# stopwords removal and lemmatization
print('Applying clean text processing step: '
    'lowercasing + punctuation removal + special character removal + '
    'lemmatizacion and stopword removal (separate column))')
stop_words = h.get_clean_stopwords(stopwords.words('english'))
train_set['qt_clean'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t))
train_set['qt_clean_stop'] = train_set.question_text.progress_apply(lambda t: h.clean_text(t, stop_words = stop_words))

# Word Frequencies by target
wf_sincere = h.word_frequency(train_set.qt_clean_stop[train_set.target==0])
wf_insincere = h.word_frequency(train_set.qt_clean_stop[train_set.target==1])
h.comparison_plot(wf_sincere[:20],wf_insincere[:20],'word','wordcount', .35, figsize = (14,8))

# Bigram Frequencies by target
stop_words = h.get_clean_stopwords(stopwords.words('english'))
wf_sincere = h.ngram_frequency(train_set.qt_clean_stop[train_set.target==0], 2)
wf_insincere = h.ngram_frequency(train_set.qt_clean_stop[train_set.target==1], 2)

h.plot_wordcloud(' '.join(wf_sincere.ngram), max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2, collocations = False)

h.plot_wordcloud(' '.join(wf_insincere.ngram), max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2, collocations = False)

h.comparison_plot(wf_sincere[:20],wf_insincere[:20],'ngram','count', .7, figsize = (14,8))

# 2.2 Document Meta features 
# These features may aid text feature modelling  
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

# outlier feature
train_set['is_outlier'] = criteria.astype('int') # binary

col = ['is_outlier', 'char_count', 'word_count', 'word_density', 'n_stopwords', 
       'n_numbers', 'n_upper', 'polarity', 'subjectivity', 'positivity',
       'noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']

# ¿Are they informative of the target? (information gain analysis)
ig_score = mutual_info_classif(train_set[col], train_set.target)
ig_score_df = pd.DataFrame({'metafeatures':col, 
                       'inf_gain':ig_score})
ig_score_df = ig_score_df.sort_values('inf_gain', ascending = False)
ig_score_df

# Most informative ngrams
ig_score_df = ig_score_df.sort_values('inf_gain', ascending = True)

plt.figure(figsize=(12,6))
plt.barh(range(len(ig_score_df.metafeatures)),ig_score_df.inf_gain, align='center', alpha=0.5)
plt.yticks(range(len(ig_score_df.metafeatures)), ig_score_df.metafeatures)
plt.xlabel('Information Gain')


# Saving Preprocessing 
with open('Data/train_preproc.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)

# 2.3 Train Valid split
# For further analysis and to avoid test leakage I will create a  
# train, test and valid set
with open('Data/train_preproc.pkl', 'rb') as input:
    train_set = pickle.load(input)


pretrain_x, test_x, pretrain_y, test_y = train_test_split(train_set, 
                                                          train_set.target,
                                                          test_size = 0.3,
                                                          stratify = train_set.target,
                                                          random_state = 33)

train_x, valid_x, train_y, valid_y = train_test_split(pretrain_x, 
                                                      pretrain_y,
                                                      test_size = 0.2,
                                                      stratify = pretrain_y,
                                                      random_state = 33)







# 5. Benchmark models ---------------------------------------------------------    
# Defining pipeline and evaluation functions
# Defining pipeline and evaluation functions
def create_pipeline(learner = None, ngram = [1,1], 
                    term_count = 1, vocabulary = None):
    estimators = []
    estimators.append(('vect', CountVectorizer(ngram_range = [1,1], 
                                               min_df = term_count,
                                               vocabulary = vocabulary)))
    estimators.append(('tfidf', TfidfTransformer()))
    
    if (learner is not None):    
        estimators.append(('learner', learner))
        
    pipe = Pipeline(estimators)
    return(pipe)

def cv_evaluation(lrn, X, y, cv = 5, cpus = 5, verbose = False, seed = 33):  
    kfold = model_selection.KFold(n_splits=cv, random_state=seed)
    results = model_selection.cross_validate(lrn, X, y, scoring = 'f1',
                                              cv=kfold, n_jobs=cpus, 
                                              verbose = verbose, 
                                              return_train_score=True)
    results = pd.DataFrame(results).mean(axis=0)
    return(results)

# Bechmark Models 
results = {}

# Naive Model
lrn = DummyClassifier(random_state = 33)
pipe_lrn = create_pipeline(lrn)
results_mod = cv_evaluation(pipe_lrn, X = train_x.question_text, y = train_y)
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

# baseline score: Logistic Regression 
lrn_basic = linear_model.LogisticRegression(class_weight = 'balanced', 
                                            C = 1e10, # Large C for no regularization
                                            random_state = 33, 
                                           penalty = 'l1')
pipe_lrn = create_pipeline(lrn_basic)
results_mod = cv_evaluation(pipe_lrn, train_x.question_text, train_y)
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

pipe_lrn.fit(train_x.question_text, train_y)
predictions_valid = pipe_lrn.predict(valid_x.question_text)
val_score = metrics.f1_score(valid_y, predictions_valid)
print('The validation F1_score was of {}'.format(val_score))

h.plot_learning_curve(pipe_lrn, text, train_y, cv=3, n_jobs=3, 
                      title = 'Learning Curves (Logistic Classifer)')    
    
# Validating preprocessing with the same base learner
results_mod = cv_evaluation(pipe_lrn, train_x.qt_clean_stop, train_y)
print('The mean train {} and test CV {}.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))










# 5. Feature Engineering ------------------------------------------------------
# 5.1 Feature Engineering: Ngram and noise reduction 
# One problem is the vast amount of text features generated in CountVectorizer
# much of them are pure noise
count_vect = CountVectorizer(binary = True)
count_vect.fit(train_x.qt_clean_stop)
xtrain_count = count_vect.transform(train_x.qt_clean_stop)
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

# ¿What if we use bigrams and trigrams? Our dimensionality increases significantly
count_vect = CountVectorizer(binary = True, ngram_range = [1,3])
count_vect.fit(train_x.qt_clean_stop)
xtrain_count = count_vect.transform(train_x.qt_clean_stop)
xtrain_count.shape
print('The DTM of bi-trigrams has {}MM terms'.format(round(xtrain_count.shape[1]/1e6,1)))

# Logistic Regression with bigram and trigrams ngrams
# Shows same performance to base test CV, but better in train CV
pipe_lrn = create_pipeline(lrn_basic, ngram =[1,3])
results_mod = cv_evaluation(pipe_lrn, train_x.qt_clean_stop, train_y)

# Performance improved by using bigram and trigrams
print('The mean train {} and test CV {} of ngrams.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))






# 5.3 Feature Engineering: Feature Selection ----------------------------------
# Another posibility is to performe feature selection on the text vectors
# Chi-Squared for Feature Selection
 
tfidf = TfidfVectorizer(ngram_range = [1,3])  
train_tfidf = tfidf.fit_transform(train_x.qt_clean_stop) 
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
train_tfidf = tfidf.fit_transform(train_x.qt_clean_stop) 

# Logistic Regression with feature selection
# Feature selection allows to reduce barely afecting baseline performance
results_mod = cv_evaluation(lrn_basic, train_tfidf, train_y)
print('The mean train {} and test CV {} of ngrams.'.format(round(results_mod['train_score'], 2),round(results_mod['test_score'], 2)))

# Saving text feature selection
with open('Data/text_feature_selection.pkl', 'wb') as output:
    pickle.dump(best_chi, output, pickle.HIGHEST_PROTOCOL)









# 6. Modelling ----------------------------------------------------------------
# 6.1 Text features Classification
# The resulting classificiation will be added to the metafeatures classification
# We will model text features selected by chi_squared
with open('Data/text_feature_selection.pkl', 'rb') as input:
    best_chi = pickle.load(input)    


# Creating train and valid vectors
pipe_vect = create_pipeline(ngram = [1,3], vocabulary = best_chi.words)
train_vect = pipe_vect.fit_transform(train_x.qt_clean_stop)
valid_vect = pipe_vect.fit_transform(valid_x.qt_clean_stop)
test_vect = pipe_vect.fit_transform(test_x.qt_clean_stop)

# Defining Models to use
log = LogisticRegression(class_weight = 'balanced', random_state = 33,
                                       solver = 'lbfgs', C = 0.5, verbose = True)
nb = naive_bayes.ComplementNB()
lgb = lightgbm.LGBMClassifier(random_state = 33, class_weight = 'balanced',
                                                objective = 'binary', n_jobs = 7,
                                                n_estimators = 1000)

models = {'logreg': log,
          'naive_bayes': nb,
          'lightgbm': lgb
          }

models['ensemble'] = VotingClassifier(estimators = [('logreg', log),
      ('nb', nb), ('lgb', lgb)], voting='soft', n_jobs = 7)

# Defining testing function
def testing_models(train_x, train_y, valid_x, valid_y, models, test_x = None, test_y = None):
    
    mod_results = {}
    
    for model in models:
        learner = models[model]
        results = {}
        print('Modelling with', model, '...')

        learner.fit(train_x, train_y)
        predictions_train = learner.predict_proba(train_x)
        predictions_valid = learner.predict_proba(valid_x)
        
        cut_off = threshold_search(train_y, predictions_train[:,1])
        pred_val_threshol = predictions_valid[:,1] > cut_off['threshold']
        
        results['train_score'] = cut_off['f1']
        results['val_score'] = metrics.f1_score(valid_y, pred_val_threshol)
        results['threshold'] = cut_off['threshold']
        results['fitted_model'] = learner
        
        if test_x is not 0 and test_y is not 0:
            print("Test data was passed, refitting model...")
            learner = models[model]
            bind_train = train_x.append(valid_x)
            bind_y = train_y.append(valid_y)
            learner.fit(bind_train, train_y)
            
            predictions_train = learner.predict_proba(bind_train)
            predictions_test = learner.predict_proba(test_x)

            cut_off = threshold_search(bind_y, predictions_train[:,1])
            pred_test_threshol = predictions_test[:,1] > cut_off['threshold']
            results['test_score'] = metrics.f1_score(test_y, pred_test_threshol)
            
        print(results)
        mod_results[model] = results   
    return(mod_results)

# Finding best text feature model
text_results = testing_models(train_vect, train_y, valid_vect, valid_y, models)

# best results is lightGBM = 'val_score': 0.5805795627859685
# Second best is Voteclassifier = 'val_score': 0.5784845076183659
# The classification result will be added as a feature with the metafeatures

# 6.2 Metafeatures classification
# This sequential classification will stack the text classification prediction
# and the 100 most importante words by feature selection

# Creating train and valid vectors
pipe_vect = create_pipeline(ngram = [1,3], vocabulary = best_chi.words)
ss_train_vect = pipe_vect.fit_transform(train_x.qt_clean_stop)
ss_valid_vect = pipe_vect.fit_transform(valid_x.qt_clean_stop)
ss_test_vect = pipe_vect.fit_transform(test_x.qt_clean_stop)

train_text_feat = ss_train_vect.todense()
train_text_feat = pd.DataFrame(train_text_feat,columns = best_chi.words)

valid_text_feat = ss_valid_vect.todense()
valid_text_feat = pd.DataFrame(valid_text_feat,columns = best_chi.words)

test_text_feat = ss_test_vect.todense()
test_text_feat = pd.DataFrame(test_text_feat,columns = best_chi.words)

# Concatenating metafeatures + top text features + prediction feature
# training
train_feat_x = pd.concat([train_x.reset_index(drop=True), 
                  train_text_feat.reset_index(drop=True)], axis=1)

#train_feat_x = train_x
lrn = text_results['lightgbm']['fitted_model']
thr = text_results['lightgbm']['threshold']
pred_train = lrn.predict_proba(train_vect)
train_feat_x['lgb_text_pred'] = (pred_train[:,1] > thr).astype('int')

# validation
val_feat_x = pd.concat([valid_x.reset_index(drop=True), 
                  valid_text_feat.reset_index(drop=True)], axis=1)

#val_feat_x = valid_x
lrn = text_results['lightgbm']['fitted_model']
thr = text_results['lightgbm']['threshold']
pred_val = lrn.predict_proba(valid_vect)
val_feat_x['lgb_text_pred'] = (pred_val[:,1] > thr).astype('int')

# test
test_feat_x = pd.concat([test_x.reset_index(drop=True), 
                  test_text_feat.reset_index(drop=True)], axis=1)

#test_feat_x = test_x
lrn = text_results['lightgbm']['fitted_model']
thr = text_results['lightgbm']['threshold']
pred_test = lrn.predict_proba(test_vect)
test_feat_x['lgb_text_pred'] = (pred_test[:,1] > thr).astype('int')

# Dropping unnecesary columns
train_feat_x = train_feat_x.drop(['qid', 'question_text', 'target', 'qt_clean', 'qt_clean_stop'], axis = 1)
val_feat_x = val_feat_x.drop(['qid', 'question_text', 'target', 'qt_clean', 'qt_clean_stop'], axis = 1)
test_feat_x = test_feat_x.drop(['qid', 'question_text', 'target', 'qt_clean', 'qt_clean_stop'], axis = 1)

# Performing Standarization
min_max_scaler = MinMaxScaler()
train_feat_x = min_max_scaler.fit_transform(train_feat_x)
val_feat_x = min_max_scaler.fit_transform(val_feat_x)

# Finding best metafeature model
lgb = lightgbm.LGBMClassifier(random_state = 33, class_weight = 'balanced',
                                                objective = 'binary', n_jobs = 7,
                                                n_estimators = 300, learning_rate=0.05)
models = {'logreg': log,
          'naive_bayes': nb,
          'lightgbm': lgb
          }

models['ensemble'] = VotingClassifier(estimators = [('logreg', log),
      ('nb', nb), ('lgb', lgb)], voting='soft', n_jobs = 7)

meta_results = testing_models(train_feat_x, train_y, val_feat_x, valid_y, models)




















