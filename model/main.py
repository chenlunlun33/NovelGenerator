import os
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from transformer import Transformer
from tools import create_tgt_mask, NovelDataset, plot_loss, collate_fn, encoder_input
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 模型參數
word_embedd_size = 9713
encoder_embed_dim = 768
decoder_embed_dim = 768
num_encoder_layers = 6
num_decoder_layers = 18
num_heads = 16
hidden_dim = 1024
dropout = 0.1
max_output_length = 50

# 模型和scaler
model = Transformer(word_embedd_size, encoder_embed_dim, decoder_embed_dim, num_encoder_layers, num_decoder_layers, num_heads, hidden_dim, dropout, max_output_length)
scaler = GradScaler()

# 設置訓練環境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 訓練參數、optimizer、criterion
embedd_vocab_path = "vocab\\vocabularyDel.txt"
batch_size = 4
num_epochs = 20
learning_rate = 0.0005
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
loss_values = []

# 獲得指定資料夾裡所有檔案路徑
folder_path = 'NovelData'
file_paths_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.relpath(os.path.join(root, file), folder_path)
        file_paths_list.append(file_path)
        
step = 0

# 訓練
def train():
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # 載入資料
        for i in file_paths_list:
            with open(folder_path + i, 'r', encoding='utf-8') as file:
                novel_data = file.read()

            # 創建Dataset、DataLoader
            dataset = NovelDataset(novel_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            
            for input_ids, target_ids in dataloader:
                input_ids = input_ids.to(device)
                # one hot encoding
                target = F.one_hot(target_ids, num_classes=word_embedd_size)
                target = target.type(torch.float64)
                target_ids = target_ids.to(device)
                # tgt_mask生成
                tgt_mask = create_tgt_mask(target)
                tgt_mask = tgt_mask.to(device)
                # pad_mask = pad_mask.to(device)
                target = target.to(device)

                # 清除梯度值
                optimizer.zero_grad()

                # 降低運算記憶體
                with autocast():
                    
                    # 前向傳播
                    output = model(input_ids, target_ids, tgt_mask=tgt_mask, pad_mask=None)
                    
                    # 計算損失
                    loss = criterion(output, target)
                    
                scaler.scale(loss).backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                num_batches += 1
            
            step += 1
            if step % 10 == 0:
                torch.save(model.state_dict(), 'save\\step' + str(step) + 'state_dict.pt')
        torch.save(model.state_dict(), 'save\\' + str(epoch) + 'state_dict.pt')

        avg_loss = total_loss / num_batches
        loss_values.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 繪製loss的摺線圖
    plot_loss(loss_values)
    
    
# 建模型
model = Transformer(word_embedd_size, encoder_embed_dim, decoder_embed_dim, num_encoder_layers, num_decoder_layers, num_heads, hidden_dim, dropout, max_output_length)
model.load_state_dict(torch.load('stepXXXstate_dict.pt'))


def evaluate():
    # 評估模式
    model.eval()

    # 讀取字彙表txt
    vocab_file = "vocab\\vocabulary.txt"

    # 創建字彙表對應token的字典
    vocab = {}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            # 移除換行符
            line = line.strip()
            token = str(index)
            word = line
            vocab[token] = word

    # 範例句
    input_word = '我是範例句我是範例句我是範例句我是範例句'

    # 轉成token
    input_tokens = encoder_input(input_word, embedd_vocab_path)
    input_tokens = np.array(input_tokens, dtype=np.int64)
    src = torch.tensor(input_tokens, dtype=torch.long)
    src = src.reshape(1, -1)

    # 需要的輸入長度
    desired_seq_length = 50

    # 檢查輸入的長度，太長截斷，太短填充
    if src.shape[1] < desired_seq_length:
        # 如果輸入長度太短，進行填充
        padding_length = desired_seq_length - src.shape[1]
        src = F.pad(src, (0, padding_length), value=0)
    elif src.shape[1] > desired_seq_length:
        # 如果輸入長度太長，進行截斷
        src = src[:, -desired_seq_length:]

    src = src.to(device)
    generate_length = 50  # 生成的最大長度

    # 起始tgt token
    start_token = torch.tensor([[0]], dtype=torch.long).to(device)

    # 生成的token sequence
    generated_sequence = [start_token]

    while len(generated_sequence) < generate_length:
        with torch.no_grad():
            # 處理生成的token序列
            decoder_input = generated_sequence[-1].long().to(device)
            generated_tokens = model.forward(src, tgt=decoder_input)

            # 處理模型的輸出預測結果
            # 將模型預測的下個token加到generated_sequence
            next_token_probs = generated_tokens[:, -1, :]
            next_token = torch.multinomial(next_token_probs, 1)
            generated_sequence.append(next_token)
            # 遇到結束符號停止模型生成
            if next_token.item() == 0:
                break

    # 將生成的token，轉為文本輸出
    generated_text = ''
    for token in generated_sequence:
        newWord = vocab[str(token.item())]
        generated_text += newWord
    print('------------------------------文章------------------------------')
    print(generated_text)