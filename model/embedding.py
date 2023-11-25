import torch.nn as nn

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_sequence_length=300):
        super(FactorizedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, input):
        embedded = self.embedding(input)
        return embedded
    
class BPE():
    
class SentencePiece():
    