#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:04:39 2021

@author: santealtamura
"""
import math
import sys
import numpy as np

def viterbi_algorithm(sentence_tokens, possible_tags, transition_matrix, 
                      emission_probabilities,
                      count_word,smoothing_strategy,
                      oneshot_words_tag_distribution):
    
    viterbi_matrix = np.zeros((len(possible_tags), len(sentence_tokens))) #matrice di viterbi
    backpointer = dict() #dizionario di dizionari
    
    #initialize first column
    for s,tag in enumerate(possible_tags):
        transition_p = transition_matrix.loc['START',tag]
        emission_p = get_emission_p(emission_probabilities, sentence_tokens[0], tag, count_word, smoothing_strategy, oneshot_words_tag_distribution, possible_tags)
        
        if transition_p == 0 : transition_p = np.finfo(float).tiny
        if emission_p == 0 : emission_p = np.finfo(float).tiny
        
        viterbi_matrix[s,0] = math.log(transition_p) +  math.log(emission_p) 
        
    #next columns
    for t in range(1,len(sentence_tokens)):
        backpointer_column = dict()
        for s, tag in enumerate(possible_tags):
            max_ , backpointer_column[s] = get_max_argmax_value(possible_tags, viterbi_matrix, transition_matrix, t, s)
            emission_p = get_emission_p(emission_probabilities, sentence_tokens[t], tag, count_word, smoothing_strategy, oneshot_words_tag_distribution, possible_tags)
            if emission_p == 0: emission_p = np.finfo(float).tiny
            viterbi_matrix[s,t] = max_ + math.log(emission_p) 
        backpointer[t] = backpointer_column   
    
    #FINAL STEP (argmax)
    max_ = -sys.maxsize
    best_path_pointer = None
    for s,tag in enumerate(possible_tags):
        end_transition = transition_matrix.loc[tag,'END']
        if end_transition == 0: end_transition = np.finfo(float).tiny
        val = viterbi_matrix[s,len(sentence_tokens) - 1] + math.log(end_transition)
        if val >= max_: max_ = val ; best_path_pointer = s
    
    #BACKTRACKING
    states = []
    states.append(best_path_pointer)
    t = len(sentence_tokens) - 1
    s = best_path_pointer
    while t >= 1:
        states.append(backpointer[t].get(s))
        s = backpointer[t].get(s)
        t = t -1
    
    #REVERSE POS TAG SEQUENCE
    pos_tags_sequence = []
    for state in list(reversed(states)): pos_tags_sequence.append(possible_tags[state])
    
    return pos_tags_sequence
 

"""Funzioni di supporto"""       
        
def get_max_argmax_value(possible_tags, viterbi_matrix, transition_matrix, t, s):
    max_ = -sys.maxsize
    argmax = None
    for s1, tag in enumerate(possible_tags):
        transition_p = transition_matrix.loc[tag,possible_tags[s]]
        if transition_p == 0 : transition_p = np.finfo(float).tiny
        val = viterbi_matrix[s1, t-1] + math.log(transition_p)
        if val >= max_: max_ = val; argmax = s1
    return max_, argmax
        
        
        
def get_emission_p(emission_probabilities, word, tag, count_word, smoothing_strategy, oneshot_words_tag_distribution, possible_tags):
    emission_p = 0
    try:
        count_word[word]
    except KeyError: #unknown_word
        emission_p = unknown_word_emission_p(smoothing_strategy, tag, possible_tags, oneshot_words_tag_distribution)         
        return emission_p
    try:
        emission_p = emission_probabilities[(word,tag)]
    except KeyError: #tag never emitted word
        emission_p = 0
    return emission_p


"""Funzioni di smoothing"""

def get_prob(tag,oneshot_words_tag_distribution):
    for tag_p,prob in oneshot_words_tag_distribution:
        if tag == tag_p:
            return prob
    
def unknown_word_emission_p(smoothing_strategy,tag,possible_tags,oneshot_words_tag_distribution):
    emission_p = 0
    if smoothing_strategy.name == 'UNKNOWN_NAME':
        if tag == 'NOUN': emission_p = 1
    if smoothing_strategy.name == 'UNKNOWN_NAME_VERB':
        if tag == 'NOUN' or tag == 'VERB': emission_p = 0.5
    if smoothing_strategy.name == 'UNKNOWN_ALL': emission_p = 1/len(possible_tags)
    if smoothing_strategy.name == 'UNKNOWN_DISTRIBUTION_ONESHOT_WORDS': 
        emission_p = get_prob(tag, oneshot_words_tag_distribution)
    return emission_p

