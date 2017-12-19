import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from sru.cuda_functional import SRU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        n, t = x.size(0), x.size(1)  # n = batch_size; t = max_text_len.
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # Reshape y.
        y = y.contiguous().view(n, t, y.size()[1])
        return y


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config
        self.num_class = config.num_class

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)
        self.tdfc = nn.Linear(config.embedding_size, 512)
        self.td = TimeDistributed(self.tdfc)
        # BatchNorm2d accepts the 1 as the num_features.
        # The expected input of BatchNorm2d is:
        # batch_size x num_features x height x width.
        # After TimeDistributed() and unsqueeze(1), the input becomes:
        # batch_size x 1 x max_text_len x embedding_size
        # The num_features here is similar to the channel in CNN.
        self.tdbn = nn.BatchNorm2d(1)

        self.rnn = nn.LSTM(input_size=512,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layers,
                          bias=True,
                          batch_first=True,
                          dropout=config.dropout_rate,
                          bidirectional=True)

        self.fc1 = nn.Linear(config.hidden_size * 2 * config.num_layers, 2048)  # x2 for bidirectional.
        self.bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, config.num_class)

    def _get_seq_len(self, x):
        """
        Args:
            x: <torch.autograd.Variable>

        Returns:
            seq_len: <list> The true length of each sequence in x.
        """
        max_seq_len = x.size()[1]
        x = x.data.cpu().numpy()
        seq_len = []
        for padded_seq in x:
            for i in list(range(max_seq_len))[::-1]:
                if padded_seq[i] != 0:
                    seq_len.append(i + 1)
                    break
        return seq_len

    def _sort_tensor(self, unsorted_tensor, seq_len):
        """
        Args:
            unsorted_tensor: <torch.autograd.Variable>
            seq_len: <list>

        Returns:
            sorted_tensor: <torch.autograd.Variable>
            seq_len: <list> Sorted lengths of tensor.
            order: <list> Sort order.
        """
        seq_len = Variable(torch.LongTensor(seq_len))
        if self.config.has_cuda:
            seq_len = seq_len.cuda()
        seq_len, order = seq_len.sort(0, descending=True)
        sorted_tensor = unsorted_tensor[order]
        # Convert to list.
        seq_len = seq_len.data.cpu().numpy().tolist()
        order = order.data.cpu().numpy().tolist()
        return sorted_tensor, seq_len, order

    def _unsort_tensor(self, sorted_tensor, order):
        """
        Args:
            sorted_tensor: <torch.autograd.Variable>
            order: <list>
        Returns:
            unsorted_tensor: <torch.autograd.Variable>
        """
        # unsorted_tensor = Variable(torch.zeros(sorted_tensor.size()))
        # if self.config.has_cuda:
        #     unsorted_tensor = sorted_tensor.cuda()
        # for i, _ in enumerate(sorted_tensor):
        #     unsorted_tensor[i, :] = sorted_tensor[order.index(i)]
        order = Variable(torch.LongTensor(order))
        if self.config.has_cuda:
            order = order.cuda()
        unsorted_tensor = sorted_tensor[order]
        return unsorted_tensor

    def forward(self, x):
        # x: batch_size x max_seq_len

        x = self.embedding(x)
        # x: batch_size x max_seq_len x embedding_dim

        x = F.relu(self.tdbn(self.td(x).unsqueeze(1))).squeeze(1)
        # x: batch_size x max_text_len x 512

        _, (h_out, _) = self.rnn(x)
        # h_out: (num_layers * num_directions) x batch_size x hidden_size

        x = h_out.transpose(0, 1).contiguous().view(self.config.batch_size, -1)
        # x: batch_size x (num_layers * num_directions) x hidden_size

        x = F.relu(self.bn(self.fc1(x)))
        # x: batch_size x 2048

        logit = self.fc2(x)

        return logit

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': self.tdfc.parameters()},
            {'params': self.tdbn.parameters()},
            {'params': self.rnn.parameters()},
            {'params': self.fc1.parameters()},
            {'params': self.bn.parameters()},
            {'params': self.fc2.parameters()}
        ], lr=lr, weight_decay=weight_decay)
