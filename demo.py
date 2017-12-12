import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_mask(sequence_lens, a, b, c):
    '''

    :param sequence_lens:  Tensor
    :param a: batch_size * num_sentences
    :param b: sequence_length
    :param c: word_hidden_size * 2
    :return:
    '''
    sequence_lens = sequence_lens.tolist()
    batch_mask = np.zeros((a, b, c), dtype=np.int64)
    i = 0
    for l in sequence_lens:
        for x in range(int(l)):
            batch_mask[i, x, :] = np.ones(c)
        i += 1
    return batch_mask.tolist()

import re
def func():
    laws = {}
    for line in open('/Users/zxsong/Downloads/BDCI2017-minglue/法律条文.txt', 'r', encoding='utf-8'):
        obj = re.findall('【(.*?)】', line)
        laws[obj[0]] = line.split('\t')[0]
        # a = input()
    for line in open('/Users/zxsong/Downloads/BDCI2017-minglue/1-train/train.txt', 'r', encoding='utf-8'):
        for law in laws:
            pattern = '.*'+str(law).strip()
            result = re.match(pattern, line)
            if result:
                print(law, laws[law])
                print(result.group())
                print(line.split('\t')[3])
                x = input()
                break

def _fetch(encoder_outputs, batch_sen_mask, length):
    result = Variable(torch.zeros(encoder_outputs.size()))
    hidden_size = encoder_outputs.size()[2]
    batch_sen_mask = torch.ByteTensor(batch_sen_mask)
    masked_select = torch.masked_select(encoder_outputs, batch_sen_mask)
    print(masked_select)
    for i in range(len(length)):
        if i == 0:
            start = 0
            end = int(length[i] * hidden_size)
        else:
            start = int(length[i-1] * hidden_size)
            if start == 0:
                start = int(length[i-2] * hidden_size)
            end = start+int(length[i] * hidden_size)
        print(start, end)
        if start != end:
            masked = masked_select[start:end].view(int(length[i]), hidden_size)
            for j in range(masked.size()[0]):
                result[i, j, :] = masked[j]

    return result
import datetime

def func2(file1, file2):
    a = set()
    _a = dict()
    b = set()
    _b = dict()
    for line in open(file1, 'r', encoding='utf-8'):
        a.add(int(line.split('\t')[0]))
        _a[int(line.split('\t')[0])] = line.split('\t')[1]
    for line in open(file2, 'r', encoding='utf-8'):
        b.add(int(line.split('\t')[0]))
        _b[int(line.split('\t')[0])] = line.split('\t')[1]

    print(len(a & b))
    c = list(a&b)
    print(_a[c[0]])
    print(_b[c[0]])
if __name__ == '__main__':
    # begin = datetime.datetime.now()
    # batch_size = 3
    # sequence_lens = 3
    # embedding_size = 2
    # length = torch.Tensor([0, 0, 2])
    # re = get_mask(length, batch_size, sequence_lens, embedding_size)
    # print(re)
    # x = torch.randn(3, 3, 2)
    # print(x)
    # masked_list = _fetch(x, re, length)
    # print(masked_list)
    # end = datetime.datetime.now()
    # print(end-begin)

    func2('/backup231/dyhu/BDCI2017-MingLue/corpus/seg_test.txt', '/disk/dyhu/Public/BDCI/data/1-train/train.txt')

    # input_size = 256
    # hidden_size = 100
    # layer = 1
    # rnn = nn.GRU(input_size, hidden_size, layer, batch_first=True, bidirectional=True)
    # input = Variable(torch.randn(4, 3, 256))
    # output, _ = rnn(input)
    # print(output.size())


