import torch
import torch.nn as nn

class FastText(nn.Module):
    """
    fastText model
    """
    def __init__(self, config):
        super(FastText, self).__init__()
        self.is_training = True
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.num_class = config.num_class
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                      embedding_dim=self.embedding_size)
        self.linear = nn.Linear(in_features=self.embedding_size, 
                                out_features=self.num_class)
   
    def forward(self, text):
        embed = self.embedding(text)
        text_embed = torch.mean(embed, dim=1)
#        print(text_embed.size())
#       text_embed = text_embed.view(-1, text_embed.size(2))
        logits = self.linear(text_embed)
        return logits    

    def get_optimizer(self, lr, lr2, weight_decay):

        return torch.optim.Adam([
            {'params': self.linear.parameters()},
            {'params': self.embedding.parameters(), 'lr': lr2}
        ], lr=lr, weight_decay=weight_decay)

