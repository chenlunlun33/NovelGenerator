import torch
import torch.nn as nn
from layers import DecoderLayer, StochasticDecoderLayer

class DecoderStochastic(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout):
        super(DecoderStochastic, self).__init__()
        self.layers = nn.ModuleList([
            StochasticDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, tgt, encoder_output, tgt_mask):
        
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask)
        
        return tgt
    
class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, encoder_output, tgt_mask):
        
        for layer in self.layers:
            tgt = layer(tgt, encoder_output, tgt_mask)
        
        return tgt