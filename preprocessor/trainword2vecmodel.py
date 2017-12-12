#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 20:48:16 2017

@author: llh
"""

import logging
import os
import multiprocessing
import argparse

from gensim.models import Word2Vec


class MySentences(object):
    def __init__(self, fname_list):
        self.fname_list = fname_list

    def __iter__(self):
        for i, fname in enumerate(self.fname_list):
            for line in open(fname, 'r'):
                line = line.strip()
                if i >= 1: # MingLueData
                    line = line.split('\t')[1]
                sentences = line.split("。")
                for sen in sentences:
                    yield sen.split()


def build_model(model_path, data_paths):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
       
#    data_paths = ['../data/seg_train_nr.txt', '../data/seg_test_nr.txt']
#    data_paths = ['../data/seg_full_shuffle_train.txt', '../data/seg_test.txt']
#    data_paths = ['../data/train_m_preprocessed.txt', '../data/test_m_preprocessed.txt']
    sentences = MySentences(data_paths) # a memory-friendly iterator
    model = Word2Vec(sentences, size=256, window=5, min_count=5, 
                         workers=multiprocessing.cpu_count(), sg=1) # sg = 1: skip-gram
    
    model.save(model_path)
    
if __name__ == "__main__":
#    model_path = './MingLueData.m.preprocessed.word2vec.128d.model'
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec-model-path", type=str)
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--test-file", type=str)
    args = parser.parse_args()
    
    data_paths = [args.train_file, args.test_file]
    build_model(args.word2vec_model_path, data_paths)
    # model = Word2Vec.load(model_path)
    # print(model.wv['抢劫'])
