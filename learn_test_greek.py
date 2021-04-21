#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:55:30 2021

@author: santealtamura
"""

import pyconll
import pandas as pd
import learning2 as learn2
import viterbi as viterbi
import time

class Smoothing:
    UNKNOWN_NAME = 1
    UNKNOWN_NAME_VERB = 2
    UNKNOWN_ALL = 3
    UNKNOWN_DISTRIBUTION_ONESHOT_WORDS = 4
    
class Language:
    GREEK = 1
    LATIN = 2


start = ['START']

pd.set_option('display.max_columns', None)

smoothing_strategy = Smoothing.UNKNOWN_DISTRIBUTION_ONESHOT_WORDS
language = Language.LATIN

if language == 1:
    train = pyconll.load_from_file('grc_perseus-ud-train.conllu')
    dev = pyconll.load_from_file('grc_perseus-ud-dev.conllu')
    test = pyconll.load_from_file('grc_perseus-ud-test.conllu')
    possible_tags = ['ADJ','ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                     'NUM', 'PART', 'PRON','SCONJ', 'VERB', 'X', 'PUNCT']
else:
    train = pyconll.load_from_file('la_llct-ud-train.conllu')
    dev = pyconll.load_from_file('la_llct-ud-test.conllu')
    test = pyconll.load_from_file('la_llct-ud-dev.conllu')
    possible_tags = ['ADJ','ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN',
                     'NUM', 'PART', 'PRON','PROPN','PUNCT', 'SCONJ', 'VERB','X']


#learning
transition_matrix = pd.DataFrame(learn2.compute_trasition_matrix(possible_tags, train), columns = list(possible_tags), index=list(possible_tags))
initial_transition_probabilities = pd.DataFrame(learn2.compute_initial_transition_probabilities(possible_tags, train), columns = list(possible_tags), index=list(start))
emission_probabilities, count_words, count_words_tag = learn2.compute_emission_probabilities(train)
oneshot_words_tag_distribution = learn2.compute_oneshot_words_distributions(possible_tags, dev)


#testing
checked_words = 0
tested_words_n = 0
start = time.time()
for sentence in test:
    pos_token_list = [token.upos for token in sentence]            
    tested_words_n = tested_words_n + len(pos_token_list)
    sentence_tokens = [token.form for token in sentence]
    result_tags = viterbi.viterbi_algorithm(sentence_tokens, possible_tags, transition_matrix, 
                                            emission_probabilities, initial_transition_probabilities,
                                            count_words, smoothing_strategy,
                                            oneshot_words_tag_distribution)
    print("=============================")
    print(result_tags)
    print()
    print(pos_token_list)
    print("=============================")
    for j in range(len(pos_token_list)):
        if pos_token_list[j] == result_tags[j]:
            checked_words = checked_words + 1
end = time.time()
print("Accuracy: ", checked_words/tested_words_n, "Time: ", end - start)
        


