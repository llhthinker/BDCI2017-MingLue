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


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.use_element = config.use_element
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=config.embedding_size, 
                                        out_channels=config.feature_size, 
                                        kernel_size=h),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=config.max_text_len-h+1))
                     for h in config.window_sizes
                    ])
        self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),
                            out_features=config.num_class)
        if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))    
    
    def forward(self, x):
        embed_x = self.embedding(x)
        
#         print(embed_x.size())
# batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
#         print(embed_x.size())
        out = [conv(embed_x) for conv in self.convs]
#         for o in out:
#             print(o.size())
        out = torch.cat(out, dim=1)
#         print(out.size(1))
        out = out.view(-1, out.size(1))
#         print(out.size())
        if not self.use_element:
            # out = F.dropout(input=out, p=self.dropout_rate)
            out = self.fc(out)
        return out

    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam([
            {'params': self.convs.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.embedding.parameters(), 'lr': lr2}
        ], lr=lr, weight_decay=weight_decay)
