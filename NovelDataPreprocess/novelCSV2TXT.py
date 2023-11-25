import pandas as pd
import os

# 獲得檔案資料夾裡的所有檔案路徑
folder_path = 'preProcessNovelData'
file_paths_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.relpath(os.path.join(root, file), folder_path)
        if isinstance(file_path, str):
            file_paths_list.append(file_path)

for i in file_paths_list:
    # 讀取csv
    df = pd.read_csv(folder_path + i)

    # 選出內文字數大於等於1000字的列
    filtered_df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x) >= 1000)]

    # 將篩選後的內容合併，以'@'作為間隔
    novel_text = '@'.join(filtered_df['text'])

    i = i.replace('.csv', '')
    
    # 儲存合併後的內容
    with open('NovelData\\' + i + '.txt', 'w', encoding='utf-8') as file:
        file.write(novel_text)