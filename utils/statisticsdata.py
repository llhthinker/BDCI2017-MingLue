#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:29:47 2017

@author: llh
"""
from itertools import groupby

def show_text_len_distribution(data):
    len_list = [len(text) for text in data]
    print(len_list[1:100])
    dic = {}
    step = 1000
    for k, g in groupby(sorted(len_list), key=lambda x: (x-1)//step):
    #    dic['{}-{}'.format(k*step+1, (k+1)*step)] = len(list(g))
        print('{}-{}'.format(k*step+1, (k+1)*step)+":"+str(len(list(g))))


def count_vocab_size(data):
    vocab_set = set()
    for text in data:
        vocab_set |= set(text)
    return len(vocab_set)

