#########################
# EDA on Train Features #
#########################

library(tidyverse)
library(data.table)
library(text2vec)
library(pbapply)
options(scipen = 99999999)

# Loading data ------------------------------------------------------------
train <- fread("Data/Train_Features.csv")
train <- train %>% # getting statistics from question with stopwords
    mutate(char_count_full = pbsapply(qt_clean, nchar),
           word_count_full = pbsapply(qt_clean, function(t) length(unlist(word_tokenizer(t)))))
colnames(train)

# Analyzing word and character frequency ----------------------------------
train %>% select(char_count:pron_count, char_count_full, word_count_full) %>% summary()

# Low count characters
train %>% filter(char_count_full == 0, target == 1) %>% View
train %>% mutate(char_count_full = pbsapply(qt_clean, function(t) nchar(t)))
