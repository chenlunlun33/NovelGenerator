import torch
import matplotlib as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        vocab = file.read().splitlines()
    return vocab

def Trans_text(text, vocab, unk_token='2'):
    decoded_text = []
    for word in text:
        if word in vocab:
            decoded_text.append(vocab.index(word))
        elif word == " ":
            decoded_text.append(1)
        else:
            decoded_text.append(2)
    return np.array(decoded_text, dtype=np.int64).flatten()

def encoder_input(src, encoder_vocab_path):

    encoder_vocab = load_vocab(encoder_vocab_path)
    encoder_encoded = Trans_text(src, encoder_vocab)

    return encoder_encoded

def decoder_input(tgt, decoder_vocab_path):
    
    decoder_vocab = load_vocab(decoder_vocab_path)
    decoder_encoded = Trans_text(tgt, decoder_vocab)

    return decoder_encoded

# 小說數據集
class NovelDataset(Dataset):
    def __init__(self, text):
        text = text + "|"
        self.textLen = len(text)
        self.count = len(text)//49
        self.src_text = text[:self.count*49]
        self.tgt_text = text[49:]
        
    def __len__(self):
        return self.count - 1

    def __getitem__(self, idx):
        src_data = self.src_text[idx*49 : (idx+1)*49] + '§'
        # '|' == 開頭結尾符號
        tgt = '|' + self.tgt_text[idx*49 : (idx+1)*49]
        # tokenizer
        input_tokens = encoder_input(src_data, embedd_vocab_path)
        target_tokens = decoder_input(tgt, embedd_vocab_path)

        input_tokens = np.array(input_tokens, dtype=np.int64)
        target_tokens = np.array(target_tokens, dtype=np.int64)
        
        return input_tokens, target_tokens

def collate_fn(data):
    input_tokens, target_tokens = zip(*data)

    # 轉換為tensor
    input_tokens = [torch.tensor(tokens, dtype=torch.long) for tokens in input_tokens]
    target_tokens = [torch.tensor(tokens, dtype=torch.long) for tokens in target_tokens]

    # 填充序列
    padded_input_tokens = pad_sequence(input_tokens, batch_first=True)
    padded_target_tokens = pad_sequence(target_tokens, batch_first=True)

    # pad_mask
    # PAD_TOKEN = 0
    # pad_mask = (padded_target_tokens != PAD_TOKEN)
    
    return padded_input_tokens, padded_target_tokens#, pad_mask

def create_tgt_mask(tgt):
    # 建立與目標序列相同形狀的全是零的張量
    tgt_mask = torch.zeros_like(tgt, dtype=torch.bool)

    # 在序列的上三角型部分設為True，表示不能看到未来資訊
    seq_len = tgt.size(1)
    tgt_mask[:, 1:seq_len] = torch.triu(torch.ones(1, seq_len-1, dtype=torch.bool), diagonal=1)

    return tgt_mask

# 繪製loss的折線圖
def plot_loss(loss_values):
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()