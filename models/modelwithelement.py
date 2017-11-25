import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

from models.fasttext import FastText
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN
from models.hierarchical import HAN
from models.cnnwithdoc2vec import CNNWithDoc2Vec
from models.rcnnwithdoc2vec import RCNNWithDoc2Vec

def _model_selector(config, model_id):
    model = None
    if model_id == 0:
        model = FastText(config)
    elif model_id == 1:
        model = TextCNN(config)
    elif model_id == 2:
        model = TextRCNN(config)
    elif model_id == 4:
        model = HAN(config)
    elif model_id == 5:
        model = CNNWithDoc2Vec(config)
    elif model_id == 6:
        model = RCNNWithDoc2Vec(config)
    else:
        print("Input ERROR!")
        exit(2)
    return model

class ElementMLP(nn.Module):
    def __init__(self, config):
        super(ElementMLP, self).__init__()

        self.element_embedding = nn.Embedding(embedding_dim=config.element_embedding_size,
                                num_embeddings=config.element_size)
        
        self.fc1 = nn.Linear(config.element_size // 2 * config.element_embedding_size,
                            config.element_size // 4 * config.element_embedding_size,)
        
        self.fc2 = nn.Linear(config.element_size // 4 * config.element_embedding_size,
                            config.element_size // 8 * config.element_embedding_size,)

        self.fc3 = nn.Linear(config.element_size // 8 * config.element_embedding_size,
                            config.element_embedding_size // 2)
    
    def forward(self, element_vector):
        embed = self.element_embedding(element_vector)
        # 32 * 17 * 256
        # print(embed.size())
        out = embed.view(embed.size(0), -1)
        # 32 * 4352
        # print(out.size())
        out = F.relu(self.fc2(F.relu(self.fc1(out))))
        out = F.relu(self.fc3(out))
        return out

class ElementCNN(nn.Module):
    def __init__(self, config):
        super(ElementCNN, self).__init__()

        self.element_embedding = nn.Embedding(embedding_dim=config.element_embedding_size,
                                num_embeddings=config.element_size)
        
        self.conv = nn.Sequential(nn.Conv1d(in_channels=config.element_embedding_size, 
                                        out_channels=config.feature_size, 
                                        kernel_size=1),
#                              nn.BatchNorm1d(num_features=config.feature_size), 
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=config.element_size//2))

        self.fc1 = nn.Linear(config.feature_size,
                            config.feature_size,)
        
        self.fc2 = nn.Linear(config.feature_size,
                            config.feature_size // 2)
    
    def forward(self, element_vector):
        embed = self.element_embedding(element_vector)
        embed = embed.permute(0, 2, 1)
        out = self.conv(embed)
        # print(out.size())
        out = out.view(-1, out.size(1))
        # print(out.size())
        # out = F.relu(self.fc2(F.relu(self.fc1(out))))
        return out


class ModelWithElement(nn.Module):
    def __init__(self, config, model_id):
        super(ModelWithElement, self).__init__()
        self.model_id = model_id
        self.nn_model = _model_selector(config, model_id)
        self.element_mlp = ElementCNN(config)
        # self.in_features = config.element_embedding_size // 2 
        self.in_features = config.feature_size
        if model_id == 1:
             self.in_features += config.feature_size*len(config.window_sizes)
        elif model_id == 2:
            self.in_features += config.feature_size*len(config.kernel_sizes)

        self.fc = nn.Linear(in_features=self.in_features,
                        out_features=config.num_class)


    def forward(self, x, element_vector):
        nn_out = self.nn_model(x)
        # print(nn_out.size())
        element_out = self.element_mlp(element_vector)
        # print(element_out.size())
        out = torch.cat((nn_out, element_out), dim=1)
        # print(out.size())
        out = self.fc(out)

        return out


    def get_optimizer(self, lr, lr2, weight_decay):
        if self.model_id == 1:
            return torch.optim.Adam([
                {'params': self.element_mlp.parameters()},
                {'params': self.nn_model.convs.parameters()},
                {'params': self.nn_model.fc.parameters()},
                {'params': self.nn_model.embedding.parameters(), 'lr': lr2}
            ], lr=lr, weight_decay=weight_decay)
        elif self.model_id == 2:
            return torch.optim.Adam([
                {'params': self.element_mlp.parameters()},
                {'params': self.bilstm.parameters()},
                {'params': self.convs.parameters()},
                {'params': self.fc.parameters()},
                {'params': self.embedding.parameters(), 'lr': lr2}
            ], lr=lr, weight_decay=weight_decay)
