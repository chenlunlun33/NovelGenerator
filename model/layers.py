import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        residual = x
        x, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(x)
        x = self.norm(x)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + self.dropout(x)
        x = self.norm(x)
        
        return x
        
class StochasticDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(StochasticDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None):
        # Self-attention
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=tgt_mask)
        x = residual + self.dropout(x)
        x = self.norm(x)
        
        # Cross-attention
        residual = x
        x, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = residual + self.dropout(x)
        x = self.norm(x)
        
        # Feed-forward + stochastic
        residual = x
        x = self.ffn(x)
        x = residual + torch.randn_like(x)
        x = self.norm(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None):
        # Self-attention
        residual = x
        x, _ = self.self_attn(x, x, x, key_padding_mask=tgt_mask)
        x = residual + self.dropout(x)
        x = self.norm1(x)

        # Cross-attention
        residual = x
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = residual + self.dropout(cross_attn_output)
        x = self.norm2(x)
        
        # Feed-forward
        residual = x
        x = self.ffn(x)
        x = residual + self.dropout(x)
        x = self.norm3(x)
        
        return x