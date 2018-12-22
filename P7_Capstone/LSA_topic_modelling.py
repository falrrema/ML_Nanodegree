#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:24:55 2018

@author: Fabs
"""

# 5.2 Feature Engineering: Topic Modelling LSA ---------------------------
# Another way to reduce dimension and noise is to perform Latent Semantic Analysis
# LSA uses truncated SVD which reduces dimensionality of the DTM matrix
# With the resulting matrices LSA can then construct topics which may be use as a feature

text = train_x.qt_clean

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
                                                          stop = 20, 
                                                          step = 2) 

x = pd.DataFrame({'Topic': range(2, 20, 2), 'Coherence': coherence_values})
ax = sns.pointplot(x = 'Topic', y = 'Coherence', data = x, linestyles=["--"]) # 2 topics

# Saving Topic Results
lda_dict = {'model_lis': model_list, 
             'coherence_values': coherence_values, 
             'topic_df': x}

with open('Data/Topic_LSA_Results.pkl', 'wb') as output:
    pickle.dump(lda_dict, output, pickle.HIGHEST_PROTOCOL)

# Inspecting 4 topics
[x.num_topics == 4 for x in model_list]
lsa_model = model_list[1]
lsa_model.print_topics(num_topics=4, num_words=20)

# Asigning topic to train_set
# Get Bigrams
train_tokens = [sentence.split() for sentence in train_set.qt_clean]
text_bigram = [' '.join(bigramer[tokens]) for tokens in tqdm(train_tokens)] # replace back

# Get Trigrams
train_tokens = [sentence.split() for sentence in text_bigram]
bi_tri_tokens = [trigramer[tokens] for tokens in tqdm(train_tokens)] 

dictionary,doc_term_matrix = h.prepare_corpus(bi_tri_tokens)
test = [h.get_topics(lsa_model, x) for x in tqdm(doc_term_matrix)]

# Saving topic feature
with open('Data/train_lsa_feat.pkl', 'wb') as output:
    pickle.dump(train_set, output, pickle.HIGHEST_PROTOCOL)
    