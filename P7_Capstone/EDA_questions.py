# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Loading libraries
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from tqdm import tqdm
from nltk import collocations as col
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Applying lowercasing + punctuation removal + stopwords removal and lemmatization
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
bigram_measures = col.BigramAssocMeasures()
doc_words = np.concatenate(np.array([sentence.split() for sentence in train_set.qt_clean_stop]))
bigramFinder = col.BigramCollocationFinder.from_words(doc_words)
bigramFinder.apply_freq_filter(20) # cutoff

# bigrams
bigram_freq = bigramFinder.ngram_fd.items()
bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
bigramPMITable.head(10)
# bigramFinder.nbest(bigram_measures.pmi, 1000)  

# Replacing collocations as bigrams in questions
collocations = set(bigramFreqTable.bigram)
train_set['qt_clean_stop_col'] = train_set.qt_clean_stop.progress_apply(lambda t: h.apply_collocations(t, collocations))

# WordCloud of collocation + sincere questions
sincere_questions = train_set[train_set.target == 0].qt_clean
insincere_questions = train_set[train_set.target == 1].qt_clean

h.plot_wordcloud(sincere_questions, max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2)

# WordCloud of collocation + insincere questions
h.plot_wordcloud(insincere_questions, max_words=70, 
max_font_size=100, stop_words = stop_words, 
figure_size=(8,10), scale = 2)














































