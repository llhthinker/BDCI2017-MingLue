import numpy as np
from collections import Counter
import random
import math

def discretization(s):
    try:
        ans = int(float(s))
        tmp = max(1, 10**(len(str(ans))-2))
        return str(ans // tmp * tmp)
    except:
        return s

def load_data(data_path):
    """
    载入数据
    """
    data= []
    labels = []
    ids = []
    max_sentence_len = 0
    
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            line_list = line.split('\t')
            one_data = line_list[1].split(' ')
            multi_label = [int(label)-1 for label in line_list[3].split(',')]
            tmp_len = len(one_data)
            if tmp_len > max_sentence_len:
                max_sentence_len = tmp_len
           # if tmp_len <= 2000:
               # data.append([discretization(i) for i in one_data])
            ids.append(line_list[0])
            data.append(one_data)
            labels.append(multi_label)
        f.close()
    print("max sentence length: ", max_sentence_len)
    return ids, data, labels


def over_sample(ids, data, labels):
    print("over sampling")
    labels_counter = Counter(labels)
    print(labels_counter)
    max_count = labels_counter.most_common(1)[0][1]
    p = dict()
    for i in labels_counter.keys():
        p[i] = (max_count - labels_counter[i]) / labels_counter[i]
    print(p)

    new_ids = []
    new_data = []
    new_labels = []
    indices = np.arange(len(labels))
    for i in indices:
        new_ids.append(ids[i])
        new_data.append(data[i])
        new_labels.append(labels[i])
        cur_p = p[labels[i]]
        if cur_p > 1:
            for c in range(int(math.sqrt(cur_p)+0.5)):
                new_ids.append(ids[i])
                new_data.append(data[i])
                new_labels.append(labels[i])
        else:
            rand_p = random.random()
            if rand_p < cur_p-0.1:
                new_ids.append(ids[i])
                new_data.append(data[i])
                new_labels.append(labels[i])

    indices = np.arange(len(new_labels))
    np.random.shuffle(indices)

    return [new_ids[i] for i in indices], np.array(new_data, dtype=np.int64)[indices], np.array(new_labels, dtype=np.int64)[indices]


def load_test_data(data_path):
    """
    载入测试数据
    """
    data= []
    tests_id = []
    max_sentence_len = 0
    with open(data_path, 'r') as f:
        for line in f:
            line_list = line.split('\t')
            tests_id.append(line_list[0])
            one_data = line_list[1].split(' ')
            tmp_len = len(one_data)
            if tmp_len > max_sentence_len:
                max_sentence_len = tmp_len
            data.append(one_data)
        f.close()
    print("max text length in test set: ", max_sentence_len)
    return tests_id, data


def build_vocabulary(data, min_count=3):
    """
    基于所有数据构建词表
    """
    count = [('<UNK>', -1), ('<PAD>', -1)]
    words = []
    for line in data:
        words.extend(line)  #TODO ?
    counter = Counter(words)
    counter_list = counter.most_common()
    for word, c in counter_list:
        if c >= min_count:
            count.append((word, c))
    dict_word2index = dict()
    for word, _ in count:
        dict_word2index[word] = len(dict_word2index)
    dict_index2word = dict(zip(dict_word2index.values(), dict_word2index.keys()))
    print("vocab size:", len(count))
    print(count[-1])
    return count, dict_word2index, dict_index2word


def build_dataset(ids, data, labels, dict_word2index, max_text_len, num_class):
    """
    基于词表构建数据集（数值化）
    """
    dataset = []
    indices = np.arange(len(labels))
#    np.random.shuffle(indices)
    new_labels = []
    new_ids = []
    for i in indices:
        new_ids.append("train_"+ids[i])
        multi_label = np.zeros(num_class)
        for li in labels[i]:
            if li > num_class:
                print(li)
                print(labels[i], ids[i])
                labels[i].remove(li)
        multi_label[labels[i]] = 1
        new_labels.append(multi_label)
        new_line = []
        for word in data[i]:
            if word in dict_word2index:
                index = dict_word2index[word]
            else:
                index = 0    # <UNK>
            new_line.append(index)

        pad_num = max_text_len - len(new_line)
        while pad_num > 0:
            new_line.append(1)   # <PAD>
            pad_num -= 1
        dataset.append(new_line[:max_text_len])
    return new_ids, np.array(dataset, dtype=np.int64), np.array(new_labels, dtype=np.int64)


def build_dataset_over_sample(ids, data, labels, dict_word2index, max_text_len):
    """
    基于词表构建数据集（数值化）
    以二分之一max_text_len进行分段
    """
    dataset = []
    indices = np.arange(len(labels))
    new_labels = []
    new_ids = []
    for i in indices:
        new_line = []
        for word in data[i]:
            if word in dict_word2index:
                index = dict_word2index[word]
            else:
                index = 0    # <UNK>
            new_line.append(index)

        pad_num = max_text_len - len(new_line)
        while pad_num > 0:
            new_line.append(1)  # <PAD>
            pad_num -= 1
        new_line_len = len(new_line)
        if new_line_len == max_text_len:
            dataset.append(new_line)
            new_ids.append("train_"+ids[i])
            new_labels.append(labels[i]-1)
        else:
            step_len = int(max_text_len)
            b = 0
            e = b + max_text_len
            while e < new_line_len:
                dataset.append(new_line[b:e])
                new_ids.append("train_"+ids[i])
                new_labels.append(labels[i]-1)
                b += step_len
                e += step_len
            pad_num = max_text_len - (new_line_len - b)
            if pad_num < int(max_text_len * 0.5):
                while pad_num > 0:
                    new_line.append(1)  # <PAD>
                    pad_num -= 1
                dataset.append(new_line[b:e])
                new_ids.append("train_"+ids[i])
                new_labels.append(labels[i]-1)
  
    new_indices = np.arange(len(new_labels))
    np.random.shuffle(new_indices)

    return [new_ids[i] for i in new_indices], np.array(dataset, dtype=np.int64)[new_indices], np.array(new_labels, dtype=np.int64)[new_indices]


def build_test_data(test_data, dict_word2index, max_text_len):
    """
    基于词表构建测试数据集（数值化）
    """
    dataset = []
    for one_data in test_data:
        new_one_data = []
        for word in one_data:

            if word in dict_word2index:
                index = dict_word2index[word]
            else:
                index = 0  # <UNK>
            new_one_data.append(index)

        pad_num = max_text_len - len(new_one_data)
        while pad_num > 0:
            new_one_data.append(1)  # <PAD>
            pad_num -= 1
        dataset.append(new_one_data[:max_text_len])

    return np.array(dataset, dtype=np.int64)


def split_data(data, radio=0.7):
    """
    将训练集分给为训练集和检验集
    """
    split_index = int(len(data) * radio)
    new_data1 = data[ : split_index]
    new_data2 = data[split_index : ]
    return new_data1, new_data2

def build_data_set_HAN(data, labels, dict_word2index, num_sentences, sequence_length, num_class):
    """
    基于词表构建数据集（数值化）
    """
    dataset = []
    indices = np.arange(len(labels))
#    np.random.shuffle(indices)
    new_labels = []
    for i in indices:
        multi_label = np.zeros(num_class)
        for li in labels[i]:
            if li > num_class:
                print(li)
                print(labels[i])
                labels[i].remove(li)
        multi_label[labels[i]] = 1
        new_labels.append(multi_label)
        # new_labels.append(labels[i]-1)
        new_line = []
        for word in data[i]:
            if word in dict_word2index:
                index = dict_word2index[word]
            else:
                index = 0    # <UNK>
            new_line.append(index)
        line_splitted = sentences_splitted(text=new_line, split_chars=[dict_word2index[split_label]
                                                                       for split_label in ['。', '！', '？']])
        # 向后补齐sequence_lengt5h
        for ls_i, ls in enumerate(line_splitted):
            line_splitted[ls_i] = sentence_padding(sentence=ls, max_length=sequence_length)
        # 向后补齐num_sentences
        pad_num = num_sentences - len(line_splitted)
        if pad_num < 0:
            line_splitted = line_splitted[-1*num_sentences:]
        while pad_num > 0:
            line_splitted.append([1 for _ in range(sequence_length)])  # <PAD>
            pad_num -= 1
        dataset.append(line_splitted)
    print(np.shape(dataset))
    # [total_size, num_sentences, sequence_length]
    return np.array(dataset, dtype=np.int64), np.array(new_labels, dtype=np.int64)



def sentence_padding(sentence, max_length):
    if len(sentence) <= max_length:
        for _ in range(max_length-len(sentence)):
            sentence.append(1)  # <PAD>
    else:
        sentence = sentence[max_length*(-1):]
    return sentence

def sentences_splitted(text, split_chars=["。"]):
    # text : list, 1-dim
    # 按照分隔符进行分句
    splitted = []
    idxs = [i for i, a in enumerate(text) if a in split_chars]
    for i, _ in enumerate(idxs):
        if i == 0:
            splitted.append(text[:idxs[i] + 1])
        else:
            splitted.append(text[idxs[i - 1] + 1: idxs[i] + 1])
    return splitted
