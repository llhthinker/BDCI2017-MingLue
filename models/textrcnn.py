import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import os

class Swish(nn.Module):

    def forward(self, input):
        return input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

class SwishSELU(nn.Module):
    alpha = 1.6732632423543772848170429916717
    
    def forward(self, input):
        return self.alpha * input * torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'


class TextRCNN(nn.Module):
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
        # batch_size x max_text_len x embedding_size
        # self.embed_bn = nn.BatchNorm1d(num_features=config.max_text_len)
        
        self.bilstm = nn.LSTM(input_size=config.embedding_size,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_layers,
                              bias=True,
                              batch_first=False,
                              dropout=config.dropout_rate,
                              bidirectional=True)

        self.convs = nn.ModuleList([
            nn.Sequential( nn.Conv1d(in_channels=config.hidden_size*2 + config.embedding_size,
                                     out_channels=config.feature_size,
                                     kernel_size=h),
#            nn.BatchNorm1d(num_features=config.feature_size),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=config.max_text_len-h+1))
            for h in config.kernel_sizes
        ])
        self.fc = nn.Linear(in_features=config.feature_size*len(config.kernel_sizes),
                            out_features=config.num_class)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))    


    def forward(self, x):
        embed_x = self.embedding(x)
#        embed_x = self.embed_bn(embed_x)
#         print(embed_x.size())
        lstm_out = self.bilstm(embed_x.permute(1,0,2))[0].permute(1,2,0)
#         # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
#         print(embed_x.size())
        x_feature = torch.cat((lstm_out, embed_x), dim=1)
        out = [conv(x_feature) for conv in self.convs]
        out = torch.cat(out, dim=1)
#         for o in out:
#         print(out.size(1))
        out = out.view(-1, out.size(1))
#         print(out.size())
#        if self.is_training:
#        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        return out

    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam([
            {'params': self.bilstm.parameters()},
            {'params': self.convs.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.embedding.parameters(), 'lr': lr2}
        ], lr=lr, weight_decay=weight_decay)
