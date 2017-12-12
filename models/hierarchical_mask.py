import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
from torch import optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordToSentence(nn.Module):
    """
    The wordacter to word-level module.
    """
    def __init__(self, config):
        super(WordToSentence, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.embedding_size)
        self.projection_nonlinearity = nn.ReLU
        self.rnn = nn.GRU
        self.word_to_sentence = self.rnn(config.embedding_size, config.word_hidden_size, bidirectional=True,
                                batch_first=True, dropout=config.dropout_rate)

        self.word_context = nn.Parameter(torch.FloatTensor(config.word_context_size, 1).uniform_(-0.1, 0.1).cuda())  # TODO 改变初始化方式
        self.word_projection = nn.Linear(config.word_hidden_size * 2, config.word_context_size)
        self.word_context_size = config.word_context_size
        self.bn = nn.BatchNorm1d(num_features=config.sequence_length)
        self.word_proj_nonlinearity = self.projection_nonlinearity()
        self.softmax = nn.Softmax(dim=1)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("pretrain...")
            self.word_embeddings.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def get_mask(self, sequence_lens, a, b, c):
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

    def _fetch(self, encoder_outputs, batch_sen_mask, length):
        '''
                encoder_outputs: Tensor
                :param batch_sen_mask: list
                :param length: tensor
                :return:
        '''
        result = Variable(torch.zeros(encoder_outputs.size())).cuda()
        hidden_size = encoder_outputs.size()[2]
        batch_sen_mask = Variable(torch.ByteTensor(batch_sen_mask)).cuda()
        masked_select = torch.masked_select(encoder_outputs, batch_sen_mask)
        # print(masked_select)
        for i in range(len(length)):
            if i == 0:
                start = 0
                end = int(length[i] * hidden_size)
            else:
                start = int(length[i - 1] * hidden_size)
                if start == 0:
                    start = int(length[i - 2] * hidden_size)
                end = start + int(length[i] * hidden_size)
            if start != end:
                masked = masked_select[start:end].view(int(length[i]), hidden_size)
                for j in range(masked.size()[0]):
                    result[i, j, :] = masked[j]

        return result

    def forward(self, x, word_hidden_stat, sequence_lens):
        '''

        :param x: [batch_size * num_sentences, sequence_length]
        :param sequence_lens: Tensor of sequences lengths of each batch element
        :return:  [batch_size * num_sentences, word_hidden_size * 2]
        '''
        word_embed = self.word_embeddings(x)
        output, _ = self.word_to_sentence(word_embed)
        mask = self.get_mask(sequence_lens, output.size()[0], output.size()[1], output.size()[2])
        output = self._fetch(output, mask, sequence_lens)
        # size: [batch_Size*num_sentences, sequence_length, word_hidden_size*2_],  e.g.: 2 3 4
        d1 = output.size()[0]
        d2 = output.size()[1]
        d3 = output.size()[2]
        projection = self.word_projection(output)
        projection = self.bn(projection)
        projection = self.word_proj_nonlinearity(projection).view(-1, self.word_context_size)  # [2x3, 5]
        attention = torch.mm(projection, self.word_context)  # [2x3, 1]
        attention = self.softmax(attention.view(d1, d2))  # [2, 3]
        attention = attention.view(1, d1 * d2).expand(d3, d1 * d2).resize(d1 * d3, d2)
        output = output.permute(2, 0, 1).resize(d1 * d3, d2)  # [4,2,3]
        sentence_tensor = (output * attention).sum(1).resize(d3, d1).transpose(0, 1)
        return sentence_tensor


class SentenceToDocment(nn.Module):
    """
    The word-to-sentence module.
    """

    def __init__(self, config):
        super(SentenceToDocment, self).__init__()
        self.projection_nonlinearity = nn.ReLU
        self.rnn = nn.GRU
        self.sentence_to_document = self.rnn(config.word_hidden_size*2, config.sentence_hidden_size,
                                   bidirectional=True, dropout=config.dropout_rate, batch_first=True)
        self.sentence_context = nn.Parameter(torch.FloatTensor(config.sentence_context_size, 1).uniform_(-0.1, 0.1).cuda())
        self.sentence_projection = nn.Linear(config.sentence_hidden_size * 2, config.sentence_context_size)
        self.sentence_context_size = config.sentence_context_size
        self.bn = nn.BatchNorm1d(num_features=config.num_sentences)
        self.sentence_proj_nonlinearity = self.projection_nonlinearity()
        self.softmax = nn.Softmax(dim=1)

    def get_mask(self, sequence_lens, a, b, c):
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

    def _fetch(self, encoder_outputs, batch_sen_mask, length):
        '''
                encoder_outputs: Tensor
                :param batch_sen_mask: list
                :param length: tensor
                :return:
        '''
        result = Variable(torch.zeros(encoder_outputs.size())).cuda()
        hidden_size = encoder_outputs.size()[2]
        batch_sen_mask = Variable(torch.ByteTensor(batch_sen_mask)).cuda()
        masked_select = torch.masked_select(encoder_outputs, batch_sen_mask)
        # print(masked_select)
        for i in range(len(length)):
            if i == 0:
                start = 0
                end = int(length[i] * hidden_size)
            else:
                start = int(length[i - 1] * hidden_size)
                if start == 0:
                    start = int(length[i - 2] * hidden_size)
                end = start + int(length[i] * hidden_size)
            if start != end:
                masked = masked_select[start:end].view(int(length[i]), hidden_size)
                for j in range(masked.size()[0]):
                    result[i, j, :] = masked[j]

        return result

    def forward(self, x, sent_hidden_stat,  num_sentences_lens):
        '''

        :param x: [batch_size, num_sentences, word_hidden_size * 2], Variable
        :param num_sentences_lens: Tensor
        :return: [batch_size, sentence_hidden_size*2]
        '''
        output, _ = self.sentence_to_document(x)
        mask = self.get_mask(num_sentences_lens, output.size()[0], output.size()[1], output.size()[2])
        output = self._fetch(output, mask, num_sentences_lens)
        d1 = output.size()[0]
        d2 = output.size()[1]
        d3 = output.size()[2]
        projection = self.sentence_proj_nonlinearity(self.bn(self.sentence_projection(output))).view(-1, self.sentence_context_size)  # [2*3, 5]
        attention = torch.mm(projection, self.sentence_context)  # [2x3, 1]
        attention = self.softmax(attention.view(d1, d2))  # [2, 3]
        attention = attention.view(1, d1 * d2).expand(d3, d1 * d2).resize(d1 * d3, d2)
        output = output.permute(2, 0, 1).resize(d1 * d3, d2)  # [4,2,3]
        document_tensor = (output * attention).sum(1).resize(d3, d1).transpose(0, 1)
        return document_tensor

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class HAN(nn.Module):

    def __init__(self, config):
        super(HAN, self).__init__()
        self.num_class = config.num_class
 #       self.dropout = nn.Dropout(p=config.dropout_rate)
        self.word_to_sentence = WordToSentence(config)
        self.sentence_to_document = SentenceToDocment(config)
        self.config = config
        self.is_training = True
        self.fc = nn.Linear(config.sentence_hidden_size * 2, self.num_class)

    def get_sequence_lens(self, x):
        sequence_lens = []
        sequence_length = x.size()[2]
        x = x.view(-1, sequence_length)
        for line in x:
            n = 0
            idx = len(line) - 1
            while idx >= 0:
                if int(line[idx]) != 1:  # <PAD>
                    break
                n += 1
                idx -= 1
            if n == len(line):
                sequence_lens.append(0)  # TODO
            else:
                sequence_lens.append((len(line) - n))
        return torch.Tensor(sequence_lens).cuda()

    def is_padded_list(self, seq):
        flag = True
        for w in seq:
            if w != 1:  # <PAD>
                flag = False
                break
        return flag

    def get_num_sentences_lens(self, x):
        '''

        :param x: batch_size, num_sentences, sequence_length], Tensor
        :return: num_sentences_lens: length: batch_size, Tensor
        '''
        num_sentences_lens = []
        for matrix in x:
            n = 0
            idx = len(matrix) - 1
            while idx >= 0:
                if not self.is_padded_list(matrix[idx]):
                    break
                n += 1
                idx -= 1
            num_sentences_lens.append((len(matrix) - n))
        return torch.Tensor(num_sentences_lens).cuda()

    def forward(self, x):
        #
        '''

        :param x: [batch_size, num_sentences, sequence_length], torch.Tensor
        :return:
        '''
        #begin = datetime.datetime.now()
        batch_size = x.size()[0]
        num_sentences = x.size()[1]
        sequence_length = x.size()[2]
        word_hidden_stat, sent_hidden_stat = self.init_rnn_hidden(batch_size)
        sequence_lens = self.get_sequence_lens(x.data)
        num_sentences_lens = self.get_num_sentences_lens(x.data)
        x = x.view(-1, sequence_length)  # [batch_size * num_sentences, sequence_length]
        x = self.word_to_sentence(x, word_hidden_stat, sequence_lens)  # [batch_size * num_sentences, word_hidden_size*2]
        x = x.resize(batch_size, num_sentences, self.config.word_hidden_size*2)  # [batch_size , num_sentences, word_hidden_size*2]
        self.document_tensor = self.sentence_to_document(x, sent_hidden_stat, num_sentences_lens)  # [batch_size, sentence_hidden_size*2]
        outputs = self.fc(self.document_tensor)
        #end = datetime.datetime.now()
        #print("model:", begin ,end, end-begin)
        return outputs

    def init_rnn_hidden(self, batch_size):
        word_hidden_stat = Variable(torch.zeros(2, batch_size*self.config.num_sentences, self.config.word_hidden_size))
        sent_hidden_stat = Variable(torch.zeros(2, batch_size, self.config.sentence_hidden_size))
        return word_hidden_stat.cuda(), sent_hidden_stat.cuda()

    # TODO
    def get_optimizer(self, lr, lr2, weight_decay):
        # for name, v in self.named_parameters():
        #     print(name, v)
        # a = input("21321321312")
        # return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        return torch.optim.Adam([
            {'params': self.word_to_sentence.word_to_sentence.parameters()},
            {'params': self.word_to_sentence.word_context},
            {'params': self.word_to_sentence.word_projection.parameters()},
            {'params': self.word_to_sentence.bn.parameters()},            
            {'params': self.word_to_sentence.word_embeddings.parameters(), 'lr': lr2},
            {'params': self.sentence_to_document.parameters()},
            {'params': self.fc.parameters()}
        #    {'params': self.intermediate_output.parameters()}
        ], lr=lr, weight_decay=weight_decay)

