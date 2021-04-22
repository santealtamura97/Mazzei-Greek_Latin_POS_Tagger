#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:35:26 2021

@author: santealtamura
"""

import numpy as np
from collections import Counter

#una_tantum -> serializzare
def compute_trasition_matrix(possible_tags,train):
    transition_matrix = np.zeros((len(possible_tags), len(possible_tags)), dtype='float32')
    for i,t1 in enumerate(possible_tags):
        for j,t2 in enumerate(possible_tags):
            transition_matrix[i][j] =  compute_transition_probability(train,t1,t2)
    return transition_matrix

  
#una_tantum -> serializzare
def compute_initial_transition_probabilities(possible_tags, train):
    initial_probabilities = np.zeros((1,len(possible_tags)), dtype='float32')
    for i,t in enumerate(possible_tags):
        initial_probabilities[0][i] = tag_initial_state_probability(train, t)
    return initial_probabilities    

#una_tantum -> serializzare
def compute_oneshot_words_distributions(possible_tags, dev):
    word_tag_set = []
    word_set = []
    for sentence in dev:
        for token in sentence:
            word_tag_set.append((token.form,token.upos))
            word_set.append(token.form)
    word_tag = dict(word_tag_set)
    count_word = dict(Counter(word_set))
    one_shot_words_tag = []
    for word in [k for k,v in count_word.items() if float(v) == 1]:
        one_shot_words_tag.append((word,word_tag[word]))
    
    tags = []
    total_tags = 0
    for word,tag in one_shot_words_tag:
        tags.append(tag)
        total_tags = total_tags + 1
    distributions = []
    for key,count in dict(Counter(tags)).items():
        distributions.append((key,count/total_tags))
    for tag in possible_tags:
        if tag not in tags:
            distributions.append((tag,0))
    
    return distributions
    
        
#una_tantum -> serializzare
def compute_emission_probabilities(train):
    word_tag_set = []
    tags_set = []
    words_set = []
    for sentence in train:
        for token in sentence:
            word_tag_set.append((token.form,token.upos))
            tags_set.append(token.upos)
            words_set.append(token.form)
            
    count_word_tag = dict(Counter(word_tag_set))
    count_tags = dict(Counter(tags_set))
    count_word = dict(Counter(words_set))
    
    emission_dict = dict()
    for key in count_word_tag:
        emission_dict[(key[0],key[1])] = count_word_tag[key]/count_tags[key[1]]

    return emission_dict,count_word,count_word_tag

#------------------------------------------------#

#t2_given_t1
def compute_transition_probability(train,tag1,tag2):
    count_t1_before_t2 = 0
    count_t1 = 0
    for sentence in train:
        for i in range (len(sentence)):
            if sentence[i-1].upos == tag1 and sentence[i].upos == tag2 and i != 0:
                count_t1_before_t2 = count_t1_before_t2 + 1
            if sentence[i].upos == tag1:
                count_t1 = count_t1 + 1
    return count_t1_before_t2/count_t1

def tag_initial_state_probability(train, tag):
    count_initial_t = 0
    for sentence in train:
        if sentence[0].upos == tag:
            count_initial_t = count_initial_t + 1
                
    return count_initial_t/len(train)





