import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


class CNNWithDoc2Vec(nn.Module):
    def __init__(self, config):
        super(CNNWithDoc2Vec, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=config.embedding_size, 
                                        out_channels=config.feature_size, 
                                        kernel_size=h),
                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=config.max_text_len-h+1))
                     for h in config.window_sizes
                    ])
        
        self.doc2vec_fc = nn.Linear(config.doc2vec_size*2, config.doc2vec_size)
        self.doc2vec_bn = nn.BatchNorm1d(config.doc2vec_size)
        self.fc = nn.Linear(in_features=config.doc2vec_size+config.feature_size*len(config.window_sizes),
                            out_features=config.num_class)
        
        if config.embedding_path:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))    
    
    def forward(self, x, doc2vec):
        embed_x = self.embedding(x)

#         print(embed_x.size())
        # batch_size,text_len, embedding_size->batch_size,embedding_size, text_len
        embed_x = embed_x.permute(0, 2, 1)
#         print(embed_x.size())
        out = [conv(embed_x) for conv in self.convs]
#         for o in out:
#             print(o.size())
        out = torch.cat(out, dim=1)
#         print(out.size(1))
        out = out.view(-1, out.size(1))

        doc2vec = F.relu(self.doc2vec_bn(self.doc2vec_fc(doc2vec)))
        out = torch.cat([out,doc2vec], dim=1)
#         print(out.size())
#        if self.is_training:
#        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        return out

    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam([
            {'params': self.convs.parameters()},
            {'params': self.fc.parameters()},
            {'params': self.embedding.parameters(), 'lr': lr2}
        ], lr=lr, weight_decay=weight_decay)
