
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class WordToSentence(nn.Module):
    """
    The wordacter to word-level module.
    """
    def __init__(self, config):
        super(WordToSentence, self).__init__()
        # wordacter embeddings
        self.word_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                            embedding_dim=config.embedding_size)
        self.projection_nonlinearity = nn.Tanh
        self.rnn = nn.GRU

        # bi-rnn
        # Inputs: wordacter embeddings
        # Outputs: word vector (word_hidden_size * 2)
        self.word_to_sentence = self.rnn(config.embedding_size, config.word_hidden_size, bidirectional=True,
                                batch_first=True)

        self.word_context = nn.Parameter(torch.FloatTensor(config.word_context_size, 1).uniform_(-0.1, 0.1).cuda())

        self.word_projection = nn.Linear(config.word_hidden_size * 2, config.word_context_size)

        # The nonlinearity to apply to the projections prior to multiplication
        # by context vector
        self.word_proj_nonlinearity = self.projection_nonlinearity()

        # Softmax layer to convert attention * projection into weights
        self.softmax = nn.Softmax()

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _sort_word_tensor(self, padded_tensor, sequence_lens):
        sequence_lens, order = sequence_lens.sort(0, descending=True)
        # print(type(sequence_lens), type(order))
        # < class 'torch.cuda.LongTensor'> < class 'torch.cuda.LongTensor' >
        padded_tensor = padded_tensor[order]
        return padded_tensor, sequence_lens, order

    def _unsort_word_tensor(self, output, order):
        '''

        :param output: <class 'torch.autograd.variable.Variable'>
        :param order:  Tensor
        :return: Variable
        '''
        padded_tensor = Variable(torch.zeros(output.size())).cuda()
        order = list(order)
        for i, _ in enumerate(output):
            padded_tensor[i, :] = (output[order.index(i)])
        return padded_tensor



    def forward(self, x, word_hidden_stat, sequence_lens):
        '''
                  [
        :param x: batch_size * num_sentences, sequence_length]
        :param sequence_lens: Tensor of sequences lengths of each batch element
        :return:  [batch_size * num_sentences, word_hidden_size * 2]
        '''
        for n, v in self.named_parameters():
            print(n ,v.grad)
        a = input("level 1")
        word_sorted, sequence_lens, order = self._sort_word_tensor(x, sequence_lens)
        word_embed = self.word_embeddings(word_sorted)
        # [batch_size * num_sentences, sequence_length, embed_size] -> [BxTx*]
        # print(type(word_embed), type(sequence_lens))
        # < class 'torch.autograd.variable.Variable'> < class 'torch.cuda.LongTensor' >
        packed = pack_padded_sequence(word_embed, list(sequence_lens), batch_first=True)
        output, _ = self.word_to_sentence(packed, word_hidden_stat)
        # output: [batch_size * num_sentences, sequence_length, word_hidden_size * 2] - > [BxTx*]
        output, _ = pad_packed_sequence(output, batch_first=True)
        # unsorted, Tensor
        output = self._unsort_word_tensor(output, order)
        # prepare final sentence_tensor :
        # [batch_size * num_sentences, word_hidden_size * 2]
        sentence_tensor = Variable(torch.zeros((output.size(0), output.size(2))).cuda())
        for word_ind in range(output.size(0)):
            projection = self.word_projection(output[word_ind])
            projection = self.word_proj_nonlinearity(projection)
            attention = torch.mm(projection, self.word_context)  # [sequence_length, 1]
            print(self.word_context)
            a = input("next word context")
            attention = self.softmax(attention.transpose(0,1))  # TODO
            sentence_tensor[word_ind, :] = output[word_ind].transpose(1, 0).mv(attention.view(-1))
            print(sentence_tensor)
        print(sentence_tensor)
        ll = input("total")
        return sentence_tensor


class SentenceToDocment(nn.Module):
    """
    The word-to-message module.
    """

    def __init__(self, config):
        super(SentenceToDocment, self).__init__()
        self.projection_nonlinearity = nn.Tanh
        self.rnn = nn.GRU
        self.sentence_to_document = self.rnn(config.word_hidden_size*2, config.sentence_hidden_size,
                                   bidirectional=True, batch_first=True)
        self.message_context = nn.Parameter(torch.FloatTensor(config.sentence_context_size, 1).uniform_(-0.1, 0.1).cuda())
        self.message_projection = nn.Linear(config.sentence_hidden_size * 2, config.sentence_context_size)
        self.message_proj_nonlinearity = self.projection_nonlinearity()
        self.softmax = nn.Softmax()

    def _sort_sentence_tensor(self, padded_tensor, num_sentences_lens):
        num_sentences_lens, order = num_sentences_lens.sort(0, descending=True)
        padded_tensor = padded_tensor[order]
        return padded_tensor, num_sentences_lens, order

    def _unsort_sentence_tensor(self, output, order):
        '''

        :param output: <class 'torch.autograd.variable.Variable'>
        :param order:  Tensor
        :return: Tensor
        '''
        padded_tensor = Variable(torch.zeros(output.size())).cuda()
        order = list(order)
        for i, _ in enumerate(output):
            padded_tensor[i,:] = (output[order.index(i)])
        return padded_tensor

    def forward(self, x, sent_hidden_stat,  num_sentences_lens):
        '''

        :param x: [batch_size, num_sentences, word_hidden_size * 2], Variable
        :param num_sentences_lens: Tensor
        :return: [batch_size, sentence_hidden_size*2]
        '''
        for n, v in self.named_parameters():
            print(n ,v.grad)
        a = input("level 2")
#        print(x)
#        ll = input("shang bian shi x in document")
        sentence_sorted, num_sentences_lens, order = self._sort_sentence_tensor(x, num_sentences_lens)
        # [batch_size, num_sentences, word_hidden_size * 2] -> [BxTx*]
#        print(sentence_sorted)
#        ll = input("sentence_sorted")
#        print(order)
#        ll = input("order")
        packed = pack_padded_sequence(sentence_sorted, list(num_sentences_lens), batch_first=True)
        output, (hidden, cell) = self.sentence_to_document(packed, sent_hidden_stat)
        # output: [batch_size, num_sentences, sentence_hidden_size * 2] - > [BxTx*]
        output, _ = pad_packed_sequence(output, batch_first=True)
#        print(output)
#        ll = input("output")
        output = self._unsort_sentence_tensor(output, order)  # Variable
#        print(output)
#        ll = input("unsort")
        # prepare final document_tensor :
        # [batch_size, num_sentences, sentence_hidden_size * 2]
        document_tensor = Variable(torch.zeros((output.size(0), output.size(2))).cuda())

        for sentence_ind in range(output.size(0)):
            projection = self.message_projection(output[sentence_ind])
            projection = self.message_proj_nonlinearity(projection)
            attention = torch.mm(projection, self.message_context)
            print(self.message_context.data)
            ll = input("sentence context")
#            print(attention)
#            print("softmax")
            attention = self.softmax(attention.transpose(1, 0))
#            print(attention)
            document_tensor[sentence_ind, :] = output[sentence_ind].transpose(1, 0).mv(attention.view(-1))
        print(document_tensor)
        ll = input("222222")
        return document_tensor

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class HAN(nn.Module):

    def __init__(self, config):
        super(HAN, self).__init__()
        self.num_class = config.num_class
        # self.dropout = nn.Dropout(p=config.dropout_rate)
        self.word_to_sentence = WordToSentence(config)
        self.sentence_to_document = SentenceToDocment(config)
        self.config = config
        self.is_training = True
        # set up the intermediate output step, if required
        # self.intermediate = False
        # self.intermediate_output_nonlinearity = nn.ELU
        # if self.intermediate:
        #     self.intermediate_output = nn.Linear(config.sentence_hidden_size * 2, config.sentence_hidden_size * 2)
        #     self.intermediate_nonlinearity = self.intermediate_output_nonlinearity()

        # final transformation to class weightings
        self.fc = nn.Linear(config.sentence_hidden_size * 2, self.num_class)

    def get_sequence_lens(self, x):
        sequence_lens = []
        sequence_length = x.size()[2]
        x = x.view(-1, sequence_length)
        for line in x:
            n = 0
            idx = len(line) - 1
            while idx >= 0:
                if int(line[idx]) != 0:
                    break
                n += 1
                idx -= 1
            if n == len(line):
                sequence_lens.append(1)  # TODO
            else:
                sequence_lens.append((len(line) - n))
        return torch.Tensor(sequence_lens).cuda()

    def is_padded_list(self, seq):
        flag = True
        for w in seq:
            if w != 0:
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
        batch_size = x.size()[0]
        num_sentences = x.size()[1]
        sequence_length = x.size()[2]
        word_hidden_stat, sent_hidden_stat = self.init_rnn_hidden(batch_size)
        # sequence_lens: 每个sequence_length的实际长度, Tensor
        print(x)
        ll = input("shang bian shi x")
        sequence_lens = self.get_sequence_lens(x.data)
        print(sequence_lens)
        ll = input("shang bian shi sequence_lens")
        num_sentences_lens = self.get_num_sentences_lens(x.data)
        print(num_sentences_lens)
        ll = input("shang bian shi num_sentences_lens")
        x = x.view(-1, sequence_length)  # [batch_size * num_sentences, sequence_length]
        x = self.word_to_sentence(x, word_hidden_stat, sequence_lens)  # [batch_size * num_sentences, word_hidden_size*2]
        x = x.view(-1, num_sentences, self.config.word_hidden_size*2)  # [batch_size , num_sentences, word_hidden_size*2]
        self.document_tensor = self.sentence_to_document(x, sent_hidden_stat, num_sentences_lens)  # [batch_size, sentence_hidden_size*2]
        # dropout or not
        # self.document_tensor = self.dropout(self.document_tensor)
        #
        # if self.intermediate:
        #     self.document_tensor = self.intermediate_output(self.document_tensor)
        #     self.document_tensor = self.intermediate_nonlinearity(self.document_tensor)

        outputs = self.fc(self.document_tensor)

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
            {'params': self.word_to_sentence.parameters()},
            {'params': self.sentence_to_document.parameters()},
            {'params': self.fc.parameters()}
        ], lr=lr, weight_decay=weight_decay)

