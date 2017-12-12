import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import os

class Inception(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert(co%4==0)
        cos=[co//4]*4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[0], 1, stride=1)),
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2)),
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            # ('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1)),
        ]))
    def forward(self,x):
       # print(x.size())
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
       # print(branch1.size())
       # print(branch2.size())
       # print(branch3.size())
       # print(branch4.size())
       # print("sss")
        result = self.activa(torch.cat((branch1, branch2,branch3, branch4), 1))
        return result
# m = torch.nn.Conv1d(256, 33, 3, stride=1)
# content_conv = nn.Sequential(
#             Inception(256, 512),
#             Inception(512, 512),
#             nn.MaxPool1d(13)
#         )
# a = Variable(torch.randn(20, 256, 13))
# print(content_conv(a).size())

class CNNwithInception(nn.Module):
    def __init__(self, config):
        super(CNNwithInception, self).__init__()
        self.is_training = True
        incept_dim = config.inception_dim  # 512
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.use_element = config.use_element
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)

        self.content_conv = nn.Sequential(
            Inception(config.embedding_size, incept_dim),
            Inception(incept_dim, incept_dim),
            nn.MaxPool1d(config.max_text_len)
        )

        # need incept_dim, linear_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(incept_dim, config.linear_hidden_size),
            nn.BatchNorm1d(config.linear_hidden_size),
            nn.ReLU(inplace=True),
            # linear_hidden_size=300  # 全连接层隐藏元数目
            nn.Linear(config.linear_hidden_size, config.num_class)
        )
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = embed_x.permute(0, 2, 1)
        # out = [conv(embed_x) for conv in self.convs]
        # out = torch.cat(out, dim=1)
        out = self.content_conv(embed_x)
        out = out.view(-1, out.size(1))
        if not self.use_element:
            out = self.fc(out)
        return out



    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam([
            {'params': self.content_conv.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.embedding.parameters(), 'lr': lr2}
        ], lr=lr, weight_decay=weight_decay)
