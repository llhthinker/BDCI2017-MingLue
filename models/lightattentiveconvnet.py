import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np


class LightAttentiveConvNet(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.num_class = config.num_class
        self.config = config
        attention_size = 10
        text_attention = nn.Parameter(torch.FloatTensor(config.embedding_size,
                                                        attention_size).uniform_(-0.1, 0.1).cuda())

        window_size = 3
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, 
                                embedding_dim=config.embedding_size)
        self.h_conv = nn.Conv1d(in_channels=config.embedding_size, 
                                out_channels=config.feature_size, 
                                kernel_size=window_size,
                                bias=False)

        self.c_conv = nn.Conv1d(in_channels=config.embedding_size,
                                out_channels=config.feature_size,
                                kernel_size=1,
                                bias=False)

        self.fc = nn.Linear(in_features=config.feature_size,
                            out_features=config.num_class)
        if config.embedding_path:
            print("Loading pretrain embedding...")
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))    
    
    def forward(self, x):
        embed_x = self.embedding(x)
        # attentive content vector generation
        attentive_content = Variable(torch.FloatTensor(embed_x.size()))
        for i in range(embed_x.size()[0]):
            hi = embed_x[i,:,:]
            # out_size:[text_len, attention_size]
            match_score = torch.mm(hi, self.text_attention)
            # [text_len, embedding_size]
            attentive_content[i,:,:] = torch.mm(F.softmax(match_score),
                                                match_score.transpose(1, 0))
#         print(embed_x.size())
        # attentive convolution
        # batch_size x text_len x embed_size  -> batch_size x embed_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        attentive_content = attentive_content.permute(0, 2, 1)
#         print(embed_x.size())
        h_out = self.h_conv(embed_x)
        h_out = F.max_pool1d(h_out)
        print(h_out.size())
        c_out = self.c_conv(attentive_content)
        c_out = F.max_pool1d(c_out)
        print(h_out.size())
        c_out = c_out.view(-1, c_out.size(1))
        out = h_out + c_out + self.tanh_bias
        out = F.tanh(out)

        out = out.view(-1, out.size(1))
#        if self.is_training:
#        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        return out

    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
