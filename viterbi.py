#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:04:39 2021

@author: santealtamura
"""
import math
import sys


def viterbi_algorithm(sentence_tokens, possible_tags, transition_matrix, 
                      emission_probabilities, initial_transition_probabilities,
                      count_word,smoothing_strategy,
                      oneshot_words_tag_distribution):
    states = []   
    for key,word in enumerate(sentence_tokens):
        p = []
        for t,tag in enumerate(possible_tags):
            emission_p = 0
            if key == 0:
                trasition_p = initial_transition_probabilities.loc['START',tag]
            else:
                trasition_p = transition_matrix.loc[states[-1]][tag]
            try:
                count_word[word]
            except KeyError: #unknown_word
                emission_p = unknown_word_emission_p(smoothing_strategy, tag, possible_tags, oneshot_words_tag_distribution)         
            emission_p = emission_probabilities.get((word,tag),emission_p)
            if emission_p != 0 and trasition_p != 0:
                state_probability = math.log(emission_p) + math.log(trasition_p)
            else:
                state_probability = -sys.maxsize
            p.append(state_probability)
        pmax = max(p)
        state_max = possible_tags[p.index(pmax)]
        states.append(state_max)
    return states

            
            
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
    if smoothing_strategy.name == 'UNKNOWN_DISTRIBUTION_ONESHOT_WORDS': emission_p = get_prob(tag, oneshot_words_tag_distribution)
                    
    return emission_p


