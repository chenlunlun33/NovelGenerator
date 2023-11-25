import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import FactorizedEmbedding
from encoder import Encoder
from decoder import Decoder, DecoderStochastic

class Transformer(nn.Module):
    def __init__(self, word_embedd_size, encoder_embed_dim, decoder_embed_dim, num_encoder_layers, num_decoder_layers, num_heads, hidden_dim, dropout, max_output_length):
        super(Transformer, self).__init__()
        self.embedding = FactorizedEmbedding(word_embedd_size, encoder_embed_dim)
        self.encoder = Encoder(num_encoder_layers, encoder_embed_dim, num_heads, hidden_dim, dropout)
        self.decoder_stochastic = DecoderStochastic(num_decoder_layers-12, decoder_embed_dim, num_heads, hidden_dim, dropout, max_output_length)
        self.decoder_1 = Decoder(6, decoder_embed_dim, num_heads, hidden_dim, dropout, max_output_length)
        self.decoder_2 = Decoder(6, decoder_embed_dim, num_heads, hidden_dim, dropout, max_output_length)
        self.fc = nn.Linear(decoder_embed_dim, word_embedd_size)
    
    def forward(self, src, tgt=None, tgt_mask=None, pad_mask=None):
        src = F.pad(src, (0, 50 - src.size(1)), value=0)
        encoder_embedded = self.embedding(src)
        encoder_output = self.encoder(encoder_embedded)
        
        if tgt is not None:
            tgt = F.pad(tgt, (0, 50 - tgt.size(1)), value=0)
            decoder_embedded = self.embedding(tgt)
            # 不同的解碼層
            decoder_output = self.decoder_1(decoder_embedded, encoder_output, tgt_mask)
            decoder_output = self.decoder_stochastic(decoder_embedded, decoder_output, tgt_mask)
            decoder_output = self.decoder_2(decoder_embedded, decoder_output, tgt_mask)
            
            # 用softmax做機率分布輸出
            output = self.fc(decoder_output)
            output = nn.functional.softmax(output, dim=-1)
        
        else:
            # 從起始token，然後逐步生成下一個token，直到生成結束或達到最大長度
            generated_tokens = []  # 用於存儲生成token
            max_length = 50  # 最大生成長度
            start_token = 0

            # 初始化生成序列，將起始token添加到生成序列中
            generated_tokens.append(start_token)
            next_token = 1

            # 逐步生成下一個token，直到達到最大長度或生成結束token
            while len(generated_tokens) < max_length and next_token != 0:
                # 使用模型解碼器來預測下一個token
                decoder_input = torch.tensor([generated_tokens[-1]], dtype=torch.long)
                decoder_embedded = self.embedding(decoder_input)
                
                # 不同的解碼層
                decoder_output = self.decoder_1(decoder_embedded, encoder_output, tgt_mask)
                decoder_output = self.decoder_stochastic(decoder_embedded, decoder_output, tgt_mask)
                decoder_output = self.decoder_2(decoder_embedded, decoder_output, tgt_mask)
                
                # 用softmax做機率分布輸出
                output = self.fc(decoder_output)
                output = nn.functional.softmax(output, dim=-1)

                # 取出最後一個時間步的機率分佈
                next_token_probs = output[0, -1, :]
                # 選擇最高機率token
                next_token = torch.argmax(next_token_probs).item()
                
                generated_tokens.append(next_token)

            return generated_tokens