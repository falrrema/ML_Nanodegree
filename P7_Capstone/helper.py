#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 18:21:15 2018

@author: Fabs
"""
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from collections import defaultdict
from tqdm import tqdm
import string as st
import pandas as pd
import matplotlib.pyplot as plt

wnl = WordNetLemmatizer()

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False, stop_words = None, scale = 1):
    ''' credits to: https://www.kaggle.com/aashita/word-clouds-of-various-shapes
    '''
    wordcloud = WordCloud(background_color='white',
                    stopwords = stop_words,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask,
                    scale = scale, 
                    collocations=True)
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

def clean_text(text, lowering = True, remove_punct = True, lemmatization = True, 
               stop_words = None):
    
    # Word tokenization
    words = text.split()
    
    # Lowercasing
    if (lowering):
        words = [word.lower() for word in words]

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

def apply_collocations(text, set_collocation):
    for b1,b2 in set_collocation:
        res = text.replace("%s %s" % (b1 ,b2), "%s_%s" % (b1 ,b2))
    return res


#def labels(from_, to_, step_):
#    return pd.Series(np.arange(from_, to_ + step_, step_)).apply(lambda x: '{:,}'.format(x)).tolist()
#
#
#def breaks(from_, to_, step_):
#    return pd.Series(np.arange(from_, to_ + step_, step_)).tolist()

