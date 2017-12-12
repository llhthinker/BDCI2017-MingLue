import json
import argparse
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str)
    parser.add_argument('-o', type=str)
    ## Count tags of money (金钱).
    parser.add_argument('--category-money-dict', type=str,
                        default='./category_money.json')
    ## Count tags of weight (重量).
    parser.add_argument('--category-weight-dict', type=str,
                        default='./category_weight.json')
    ## Count tags of blood alcohol concentration (血液酒精浓度).
    parser.add_argument('--category-BAC-dict', type=str,
                        default='./category_BAC.json')
    args = parser.parse_args()


    category_dict = [args.category_money_dict,
                     args.category_weight_dict,
                     args.category_BAC_dict]
    for cd in category_dict:
        with open(cd, 'r') as f:
            category = list(json.loads(f.read()).keys())

        # 计数置零。
        category_count = {}
        for c in category:
            category_count[c] = 0

        penalty_dict = {}
        for i in ['1','2','3','4','5','6','7','8']:
            penalty_dict[i] = category_count.copy()

        with open(args.i, 'r') as f:
            for line in f.readlines():
                text, penalty = line.split('\t')[1:3]
                for c in category:
