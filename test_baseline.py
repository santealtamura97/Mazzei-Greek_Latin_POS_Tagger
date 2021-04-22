#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 17:12:42 2021

@author: santealtamura
"""

import learning2 as learn
import pyconll
import time
from collections import Counter
import enum

class Language(enum.Enum):
    GREEK = 1
    LATIN = 2

language = Language.GREEK

if language.name == 'GREEK':
    train = pyconll.load_from_file('grc_perseus-ud-train.conllu')
    test = pyconll.load_from_file('grc_perseus-ud-test.conllu')
    possible_tags = ['ADJ','ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                 'NUM', 'PART', 'PRON','SCONJ', 'VERB', 'X', 'PUNCT']
elif language.name == 'LATIN':
    train = pyconll.load_from_file('la_llct-ud-train.conllu')
    test = pyconll.load_from_file('la_llct-ud-dev.conllu')
    possible_tags = ['ADJ','ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'NOUN',
                     'NUM', 'PART', 'PRON','PROPN', 'PUNCT', 'SCONJ', 'VERB', 'X']

count_words_tag = learn.compute_emission_probabilities(train)[2]

def baseline_algorithm(sentence_tokens,count_words_tag,possible_tags):
    tags = []
    for word in sentence_tokens:
        tag_max = 'NOUN'
        count_max_tag = 0
        for tag in possible_tags:
            if count_words_tag.get((word,tag),0) > count_max_tag:
                count_max_tag = count_words_tag[word,tag]
                tag_max = tag    
        tags.append(tag_max)
    return tags

#testing
checked_words = 0
tested_words_n = 0
error_list = []
start = time.time()
for sentence in test:
    pos_token_list = [token.upos for token in sentence]
    sentence_tokens = [token.form for token in sentence]
    tested_words_n = tested_words_n + len(pos_token_list)
    result_tags = baseline_algorithm(sentence_tokens, count_words_tag, possible_tags)
    for j in range(len(pos_token_list)):
        if pos_token_list[j] == result_tags[j]:
            checked_words = checked_words + 1     
        else:
            error_list.append(pos_token_list[j])
end = time.time()

print("Algoritmo: BASELINE")
print("Linguaggio testato: ", language.name)
print("Pos Tag corretti: ", checked_words)
print("Pos Tag sbagliati: ", tested_words_n - checked_words)
print("Totale parole valutate: ",tested_words_n)
print("Accuratezza: ", format((checked_words/tested_words_n)*100,'.2f'),"%")
print("Conteggi errori: ", dict(Counter(error_list)))
print("Tempo di esecuzione: ", format(end - start,'.2f')," sec")
