import re
import pandas as pd
import os

def preprocess_text(content):
    
    if type(content) != str:
        return ''

    # 將連續14個空格轉換成"@"
    content = re.sub(r' {14}', '@', content)

    # 刪除網址
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    # 多餘文字
    content = re.sub('多於文字', '', content)
    
    # 去除無法解碼的字元
    content = content.encode('utf-8', errors='ignore').decode('utf-8')
    
    return content

# 抓取路徑
folder_path = 'NovelData'
file_paths_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.relpath(os.path.join(root, file), folder_path)
        file_paths_list.append(file_path)

for i in file_paths_list:
    novel_data = pd.read_csv(folder_path + '\\' + i)
    for j in range(len(novel_data)):
        outline = novel_data['text'][j]
        # 執行預處理
        novel_data['text'][j] = preprocess_text(outline)
    novel_data.to_csv('preProcessNovelData\\' + i)
    