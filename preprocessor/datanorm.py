#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 17-10-19 下午5:35
@Author  : szxSpark
@Email   : szx_spark@outlook.com 1800
@File    : seg.py.py
@Software: PyCharm Community Edition
"""
import jieba.posseg as pseg
import jieba
import random
import datetime
import pickle
import logging
from gensim import corpora
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
import json
import jieba.posseg as pseg
# stop_flag = ['c', 'u', 't', 'p', 'f', 'm', 'r', 'nr']
# stop_flag = ['u', 'p', 'f', 'r', 'nr']
stop_flag = ['nr']
# c:连词 u：助词 t:时间词 p:介词 f:方位词 m:数词 r:代词 nr:人名
# u p f r nr

def  load_train_data(filename):
    # out_filename = filename[0:-4]+"_seg.txt"  # ../data/train_seg.txt
    out_filename = "../data/seg_train_nr.txt"
    num = 0
    starttime = datetime.datetime.now()
    seg_list = []
    label_list = []

    print("Begin preprocessing:", starttime)
    with open(out_filename, 'w', encoding="utf-8") as f:
        for line in open(filename, 'r', encoding="utf-8"):
            new_str = []
            num += 1
            print("preprocessing: sentence", num)
            temp = line.strip().split("\t")
            text = temp[1].strip()
            temp_list = list(pseg.cut(text))
            new_tmp = []
            for w in temp_list:
                word = w.word
                flag = w.flag
                if flag not in stop_flag:
                    new_tmp.append(word)
                else:
                    new_tmp.append("</nr>")
            new_str.append(temp[0].strip())
            new_str.append(" ".join(new_tmp))
            new_str.append(temp[2].strip())
            new_str.append(temp[3].strip())
            f.write("\t".join(new_str).strip()+"\n")
            seg_list.append(new_tmp)
            label_list.append(int(temp[2]))
    endtime = datetime.datetime.now()
    print("End preprocessing:", endtime)
    print("Time used:", endtime-starttime)
    print(len(seg_list), len(label_list))
    # with open('../pickles/seg_list.pickle', 'wb') as f:
    #     pickle.dump(seg_list, f)
    # with open('../pickles/label_list.pickle', 'wb') as f:
    #     pickle.dump(label_list, f)

def load_test_data(filename):
    # out_filename = filename[0:-4]+"_seg.txt"  # ../data/train_seg.txt
    out_filename = "../data/seg_test_nr.txt"
    num = 0
    starttime = datetime.datetime.now()
    seg_list = []
    id_list = []
    print("Begin preprocessing:", starttime)
    with open(out_filename, 'w', encoding="utf-8") as f:
        for line in open(filename, 'r', encoding="utf-8"):
            new_str = []
            num += 1
            print("preprocessing: sentence", num)
            temp = line.strip().split("\t")
            temp_list = list(pseg.cut(temp[1].strip()))
            new_tmp = []
            for w in temp_list:
                word = w.word
                flag = w.flag
                if flag not in stop_flag:
                    new_tmp.append(word)
                else:
                    new_tmp.append("</nr>")
            new_str.append(temp[0].strip())
            new_str.append(" ".join(new_tmp))
            f.write("\t".join(new_str).strip()+"\n")
            seg_list.append(new_tmp)
    endtime = datetime.datetime.now()
    print("End preprocessing:", endtime)
    print("Time used:", endtime-starttime)
    print(len(seg_list))
    # with open('../pickles/test_seg_list.pickle', 'wb') as f:
    #     pickle.dump(seg_list, f)
    # with open('../pickles/test_id_list.pickle', 'wb') as f:
    #     pickle.dump(id_list, f)

def data_sorted():
    '''
     按照每篇文章包含的句子数量， 由小到大排序
    :return:
    '''
    # TODO 分割句子
    with open('../pickles/seg_list.pickle', 'rb') as f:
        sentences = pickle.load(f)
    with open('../pickles/label_list.pickle', 'rb') as f:
        label_list = pickle.load(f)
    print(sentences[-1])
    print(label_list[-1])
    temp_dict = {}
    for i in range(len(sentences)):
        sentence = sentences[i]
        count = sentence.count("。")
        temp_dict[i] = count
        print(i, count)
    temp_tuple = sorted(temp_dict.items(), key=lambda asd: asd[1], reverse=False)
    sentences_sorted = []
    label_sorted = []
    f = open("../data/train_sorted.txt", "w", encoding="utf-8")
    for t in temp_tuple:
        print(str(t))
        sentences_sorted.append(sentences[t[0]])
        f.write(" ".join(sentences[t[0]]).strip())
        label_sorted.append(label_list[t[0]])
        f.write("\t"+str(label_list[t[0]]))
        f.write("\n")
    f.close()

    print(len(sentences_sorted))
    print(len(label_sorted))

    print(sentences[-1])
    print(label_list[-1])

    with open('../pickles/seg_sorted_list.pickle', 'wb') as f:
        pickle.dump(sentences_sorted, f)
    with open('../pickles/label_sorted_list.pickle', 'wb') as f:
        pickle.dump(label_sorted, f)

def clean(number):
    '''
        过滤不合格式的文章
    '''
    num = 0
    k = 0
    with open("../data/clean_train_1.0.txt", "w", encoding="utf-8") as f:
        for line in open("../data/train.txt"):
            num += 1
            # print("preprocessing: sentence", num)
            temp = line.strip().split("\t")
            text = temp[1].strip()
            if text[0] == "�" and len(text) <= number:
                print(num, "wrong", text.count("。"), len(text))
            else:
                if text[0] == "�":
                    print(num, "True", text.count("。"), len(text))
                k += 1
                f.write(line)

    print("clean_train:", k)

def split_train_dev(ratio):
    '''

    :param ratio: 80%
    :return:
    '''
    with open("../pickles/seg_sorted_list.pickle", "rb") as f:
        sentences = pickle.load(f)
    with open("../pickles/label_sorted_list.pickle", "rb") as f:
        label_list_sorted = pickle.load(f)

    total_number = len(sentences)  # 39969
    train_size = int(ratio * total_number)
    print("total_number:", total_number)
    print("train_size:", train_size)

    # 在A - B的闭区间生成COUNT个不重复的数字
    A = 0
    B = total_number - 1
    count = train_size
    resultList = random.sample(range(A, B + 1), count)
    train_seg_list = []
    train_label_list = []
    dev_seg_list = []
    dev_label_list = []
    for i in range(len(sentences)):
        if i in resultList:
            train_seg_list.append(sentences[i])
            train_label_list.append(label_list_sorted[i])
        else:
            dev_seg_list.append(sentences[i])
            dev_label_list.append(label_list_sorted[i])
    print("train_data:")
    print(train_seg_list[0])
    print(train_label_list[0])
    print("dev_data:")
    print(dev_seg_list[0])
    print(dev_label_list[0])
    with open("../pickles/train_seg_list.pickle", "wb")as f:
        pickle.dump(train_seg_list, f)
        print(len(train_seg_list))
        # 31975

    with open("../pickles/dev_seg_list.pickle", "wb")as f:
        pickle.dump(dev_seg_list, f)
        print(len(dev_seg_list))
        # 7994

    with open("../pickles/train_label_list.pickle", "wb")as f:
        pickle.dump(train_label_list, f)
        print(len(train_label_list))
        # 31975

    with open("../pickles/dev_label_list.pickle", "wb")as f:
        pickle.dump(dev_label_list, f)
        print(len(dev_label_list))
        # 7994



no_below = 0
no_above = 1
embedding_size = 200

def write_token2id(dictionary):
    f = open('token2id.txt', 'w')
    dict1 = dict(dictionary.token2id)
    dict2 = dict((value, key) for key,value in dict1.items())
    dict_list = sorted(dict2.items(), key=lambda asd:asd[0], reverse=False)
    for r in dict_list:
        f.write(str(r)+'\n')
    f.close()

def write_dfs(dictionary):
    #单词t出现在几篇文档中
    f = open('dfs.txt', 'w')
    dict_list = sorted(dictionary.dfs.items(), key=lambda asd: asd[1], reverse=True)
    for r in dict_list:
        t = (dictionary[r[0]], r[1])
        result = json.dumps(t, ensure_ascii=False)
        f.write(result + "\n")

def write_dictionary():
    sentences = []
    with open("../pickles/seg_list.pickle", "rb") as f:
        temp1 = pickle.load(f)
    with open("../pickles/test_seg_list.pickle", "rb") as f:
        temp2 = pickle.load(f)
    sentences.extend(temp1)
    sentences.extend(temp2)
    dictionary = corpora.Dictionary(sentences)
    print(len(dictionary.token2id)) # 485329
    print(dictionary.num_docs)  # 69969
    dictionary.filter_extremes(no_below=2, no_above=0.97, keep_n=100000)
    print("After filter_extremes:")
    print(len(dictionary.token2id))  # 100000
    print(dictionary.num_docs)  # 69969
    # dictionary.save("../pickles/dictionary.dic")
    token2id = dict(dictionary.token2id)
    id = token2id["�"]  # 23
    print(id)  # 87800
    del token2id["�"]
    if "UNK" not in token2id:
        token2id["UNK"] = id
    print(token2id["UNK"])  # 87800
    write_token2id(dictionary)
    write_dfs(dictionary)
    with open("../pickles/token2id.pickle", "wb")as f:
        pickle.dump(token2id, f)

if __name__ == "__main__":
    # clean(110)
    # load_train_data("../data/clean_train_1.0.txt")
    # load_test_data("../data/test.txt")
    # data_sorted()
    # split_train_dev(0.8)  # right
    # write_dictionary()  # 添加UNK
# ------------------------------------------------------
    load_train_data("../data/1-train/train.txt")
    load_test_data("../data/2-test/test.txt")
