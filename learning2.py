#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:35:26 2021

@author: santealtamura
"""

import numpy as np
from collections import Counter
from conllu import parse_incr


def compute_transition_matrix(possible_tags, train):
    transition_matrix = np.zeros((len(possible_tags), len(possible_tags)), dtype='float32')
    
    transition_counter_dict = dict()
    counter_dict = dict()
    count_initial_dict = dict()
    
    for tag1 in possible_tags:
        counter_dict[tag1] = 0
        count_initial_dict[tag1] = 0
        for tag2 in possible_tags:
            transition_counter_dict[(tag1, tag2)] = 0
    
    sentence_n = 0
    for sentence in parse_incr(train):
        sentence_n += 1
        for i in range(len(sentence)):
            word_before = sentence[i-1]
            word = sentence[i]
            if i == 0:
                if word["upos"] in count_initial_dict.keys():
                    count_initial_dict[word["upos"]] = count_initial_dict[word["upos"]] + 1
            if (word_before["upos"], word["upos"]) in transition_counter_dict.keys() and i != 0:
                transition_counter_dict[(word_before["upos"], word["upos"])] = transition_counter_dict[(word_before["upos"], word["upos"])] + 1
            if word["upos"] in counter_dict.keys():
                counter_dict[word["upos"]] = counter_dict[word["upos"]] + 1
            if i == len(sentence) - 1:
                if (word["upos"], 'END') in transition_counter_dict.keys():
                    transition_counter_dict[(word["upos"], 'END')] = transition_counter_dict[(word_before["upos"], word["upos"])] + 1
    
    #probabilità di transizione iniziali           
    for i,t in enumerate(possible_tags):
        transition_matrix[0][i] = count_initial_dict[t]/sentence_n
    #probabilità di transizione intermedie
    for i,t1 in enumerate(possible_tags):
        for j,t2 in enumerate(possible_tags):
            if i >= 1 and j >= 1 and i < (len(possible_tags) - 1):
                transition_matrix[i][j] =  transition_counter_dict[(t1,t2)]/counter_dict[t1]
    
    train.seek(0)
    
    return transition_matrix

#una_tantum -> serializzare
def compute_oneshot_words_distributions(possible_tags, dev):
    word_tag_set = []
    word_set = []
    for sentence in parse_incr(dev):
        for token in sentence:
            word_tag_set.append((token["form"],token["upos"]))
            word_set.append(token["form"])
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
    for sentence in parse_incr(train):
        for token in sentence:
            word_tag_set.append((token["form"],token["upos"]))
            tags_set.append(token["upos"])
            words_set.append(token["form"])
            
    count_word_tag = dict(Counter(word_tag_set))
    count_tags = dict(Counter(tags_set))
    count_word = dict(Counter(words_set))
    
    emission_dict = dict()
    for key in count_word_tag:
        emission_dict[(key[0],key[1])] = count_word_tag[key]/count_tags[key[1]]

    return emission_dict,count_word,count_word_tag

#------------------------------------------------#



