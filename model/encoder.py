import torch.nn as nn
from layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x