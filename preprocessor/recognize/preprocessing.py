import os
import sys
import re
import json
import multiprocessing
import time
import argparse
from tqdm import tqdm
import jieba
import jieba.posseg

import toolkit
import recognize

def text2seq(text, args):
    text = re.sub(r'/s', '', text)  # Remove all spaces in the text.
    if args.date:
        text = recognize.recognize_date(text)
    if args.money:
        text = recognize.recognize_money(text, args.category_money_dict)
    if args.weight:
        text = recognize.recognize_weight(text, args.category_weight_dict)
    if args.BAC:
        text = recognize.recognize_BAC(text, args.category_BAC_dict)
    # Replace categories with aliases, in case of being segmented.
    for category in args.category_alias_dict:
        text = text.replace(category, args.category_alias_dict[category])
    if args.nr or args.ns or args.m:
        seq = []
        pair = jieba.posseg.cut(text)
        for word, flag in pair:
            if args.nr and flag == 'nr':
                seq.append('[nr]')
            elif args.ns and flag == 'ns':
                seq.append('[ns]')
            elif args.m and flag == 'm':
                seq.append('[m]')
            else:
                seq.append(word)
    else:
        seq = list(jieba.cut(text))
    # Recover categories from aliases.
    alias_category_dict = {a: c for c, a in args.category_alias_dict.items()}
    for i in range(len(seq)):
        if seq[i] in alias_category_dict:
            seq[i] = alias_category_dict[seq[i]]
    return seq

def line_preprocessing(line, args):
    content = line.split('\t')
    seq = text2seq(content[1], args)
    content[1] = ' '.join(seq)
    return '\t'.join(content)

def _line_preprocessing(param):
    return line_preprocessing(*param)

def preprocessing(input_file, output_file, core_num, args):
    with open(input_file, 'r') as f:
        corpus = f.read().strip()
    # Convert some Chinese characters from full forms into half forms.
    corpus = toolkit.full2half(corpus)
    line = corpus.split('\n')
    pool = multiprocessing.Pool(core_num)

    param = [[l, args] for l in line]
    line_processed = pool.map(_line_preprocessing, param)
    with open(output_file, 'w') as f:
        f.write('\n'.join(line_processed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic settings.
    parser.add_argument('--input-file', type=str)
    parser.add_argument('--output-file', type=str, default='./output.txt')
    parser.add_argument('--core-num', type=int, default=multiprocessing.cpu_count())
    # Recognize and discrete settings.
    ## Recognize all person names (人名) with [nr].
    parser.add_argument('--nr', dest='nr', action='store_true')
    parser.add_argument('--no-nr', dest='nr', action='store_false')
    parser.set_defaults(nr=True)
    ## Recognize all place names (地名) with [ns].
    parser.add_argument('--ns', dest='ns', action='store_true')
    parser.add_argument('--no-ns', dest='ns', action='store_false')
    parser.set_defaults(ns=True)
    ## Recognize all numbers (数字) with [m].
    parser.add_argument('--m', dest='m', action='store_true')
    parser.add_argument('--no-m', dest='m', action='store_false')
    parser.set_defaults(m=True)
    ## Recognize all dates (日期) with [date].
    parser.add_argument('--date', dest='date', action='store_true')
    parser.add_argument('--no-date', dest='date', action='store_false')
    parser.set_defaults(date=True)
   ## Recognize and discrete all money (金钱).
    parser.add_argument('--money', dest='money', action='store_true')
    parser.add_argument('--no-money', dest='money', action='store_false')
    parser.set_defaults(money=True)
    parser.add_argument('--category-money-dict', type=str,
                        default='./category_money.json')
    ## Recognize and discrete all weight (重量).
    parser.add_argument('--weight', dest='weight', action='store_true')
    parser.add_argument('--no-weight', dest='weight', action='store_false')
    parser.set_defaults(weight=True)
    parser.add_argument('--category-weight-dict', type=str,
                        default='./category_weight.json')
    ## Recognize and discrete all blood alcohol concentration (血液酒精浓度).
    parser.add_argument('--BAC', dest='BAC', action='store_true')
    parser.add_argument('--no-BAC', dest='BAC', action='store_false')
    parser.set_defaults(BAC=True)
    parser.add_argument('--category-BAC-dict', type=str,
                        default='./category_BAC.json')
    # Others.
    parser.add_argument('--category-alias-dict')
    args = parser.parse_args()

    category = []
    if args.date:
        category += ['[date]']
    if args.money:
        with open(args.category_money_dict, 'r') as f:
            args.category_money_dict = json.loads(f.read())
        category += list(args.category_money_dict.keys())

    if args.weight:
        with open(args.category_weight_dict, 'r') as f:
            args.category_weight_dict = json.loads(f.read())
        category += list(args.category_weight_dict.keys())

    if args.BAC:
        with open(args.category_BAC_dict, 'r') as f:
            args.category_BAC_dict = json.loads(f.read())
        category += list(args.category_BAC_dict.keys())
    args.category_alias_dict = toolkit.alias(category)

    jieba.initialize()

    time_start = time.time()
    preprocessing(args.input_file, args.output_file, args.core_num, args)
    time_end = time.time()
    print('Time consumed: {:.2f}s'.format(time_end - time_start))
