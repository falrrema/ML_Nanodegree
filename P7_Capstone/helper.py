#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 18:21:15 2018

@author: Fabs
"""
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from wordcloud import WordCloud
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import learning_curve
import string as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns


wnl = WordNetLemmatizer()

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, 
                   figure_size=(24.0,16.0), title = None, title_size=40, 
                   image_color=False, stop_words = None, scale = 1, 
                   collocations = True):
    ''' credits to: https://www.kaggle.com/aashita/word-clouds-of-various-shapes
    '''
    wordcloud = WordCloud(background_color='white',
                    stopwords = stop_words,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask,
                    scale = scale, 
                    collocations=collocations)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

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
    
    # Stopwords removal
    if (stop_words is not None):
        words = [word for word in words if word not in stop_words] 
        
    # Lemmatization
    if (lemmatization):    
        words = [wnl.lemmatize(word, pos='v') for word in words]  
    
    # Words to sentence
    sentence = " ".join(words)
    return sentence

def get_clean_stopwords(stop_words):
    words = [word.lower() for word in stop_words]
    translator = str.maketrans('', '', st.punctuation)
    words = [word.translate(translator) for word in words]
    return(words)

def word_frequency(text):
    freq_dict = defaultdict(int)
    for sentences in tqdm(text):
        for word in sentences.split():
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    return(fd_sorted)

def ngram_tokens(sentence, ngram = 2):
    words = sentence.split()
    return(list(ngrams(words, ngram)))

def ngram_frequency(text, ngram = 2):
    freq_dict = defaultdict(int)
    for sentences in tqdm(text):
        ng_tokens = ngram_tokens(sentences, ngram)
        for ng in ng_tokens:
            index = '_'.join(ng)
            freq_dict[index] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["ngram", "count"]
    return(fd_sorted)

def apply_collocations(text, set_collocation):
    for b1,b2 in set_collocation:
        res = text.replace("%s %s" % (b1 ,b2), "%s_%s" % (b1 ,b2))
    return res

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, scoring = 'f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def comparison_plot(df_1,df_2,col_1,col_2, space, figsize = (10,8), color = 'salmon'):
    ''' Credits to https://www.kaggle.com/arunsankar/key-insights-from-quora-insincere-questions
    '''
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color=color)
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color=color)

    ax[0].set_xlabel('Word count', size=14, color="black")
    ax[0].set_ylabel('Words', size=14, color="black")
    ax[0].set_title('Top words in sincere questions', size=18, color="black")

    ax[1].set_xlabel('Word count', size=14, color="black")
    ax[1].set_ylabel('Words', size=14, color="black")
    ax[1].set_title('Top words in insincere questions', size=18, color="black")

    fig.subplots_adjust(wspace=space)
    plt.show()

# Getting Concept words by collocations
def get_collocations(text, verbose = True, bigram_freq = True):
    if (verbose):
        print('Word Tokenization...')
    tokens = [t.split() for t in text]
    
    if (verbose):
        print('Making Bigramer Model...')
        
    bigramer = Phrases(tokens)  # train model with default settings
    
    
    if (bigram_freq):   
        if (verbose):
            print('Making Bigramer list...')
        
        bigram_counter = list()
        bigram_list = list(bigramer.vocab.items())
        for key, value in bigram_list:
            str_key = key.decode()
            if len(str_key.split("_")) > 1:
                bigram_counter.append(tuple([str_key, value]))
        bigram_df = pd.DataFrame(bigram_counter, columns=['bigrams', 'count'])

    # Replacing collocations in questions to improve interpretability
    bigramer = Phraser(bigramer) # faster implementation
    #train_set['qt_clean_col'] = [' '.join(bigramer[tokens]) for tokens in tqdm(train_tokens)]
    
    if (bigram_freq):
        res_dict = {'bigramer': bigramer, 'bigram_freq': bigram_df}
    else:
        res_dict = {'bigramer': bigramer, 'bigram_freq': None}
    
    return(res_dict)

#def labels(from_, to_, step_):
#    return pd.Series(np.arange(from_, to_ + step_, step_)).apply(lambda x: '{:,}'.format(x)).tolist()
#
#
#def breaks(from_, to_, step_):
#    return pd.Series(np.arange(from_, to_ + step_, step_)).tolist()

