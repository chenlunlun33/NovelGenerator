import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from opencc import OpenCC
import time

# 簡體轉繁體
cc = OpenCC('tw2sp')

label = []
text = []
headers = {'user-agent': 'Mozilla/5.0'}

def get_webpage_text(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    time.sleep(0.05)
    text = soup.get_text()
    return text

# 設定要爬取的網頁 URL
# url = 'http://'

# ll = [
#     ['https://', '名子',首頁編號, 尾頁編號],
# ]


for a in range(len(ll)):
    text = []
    for i in range(ll[a][2], ll[a][3]):
        # 取得網頁內文
        url = ll[a][0] + str(i) + '.html'
        print(url)
        webpage_text = get_webpage_text(url)
        
        # 假設你要找的部分被 start_word 和 end_word 兩個字詞包圍
        start_word = ""
        end_word = ""
        # 使用正則表達式進行匹配
        pattern = r"{}(.*?){}".format(re.escape(start_word), re.escape(end_word))
        match = re.search(pattern, webpage_text, re.DOTALL)

        if match:
            target_text = match.group(1)
            # print(target_text)
        
        # 將文本按行分割成列
        lines = target_text.split("\n")

        # 使用列過濾掉空白行
        filtered_lines = [line.strip() for line in lines if line.strip()]
        # print(filtered_lines)
        
        # 將過濾後的行重新組合成文本
        filtered_text = "\n".join(filtered_lines[:])
        text.append(filtered_text)

        # label.append(cc.convert(filtered_label))
        # text.append(cc.convert(filtered_text))
        
    # # 輸出結果
    # print(filtered_label)
    # print(filtered_text)

    # 儲存csv
    df = pd.DataFrame((text), columns = ['text'])
    df.to_csv('NovelData\\' + ll[a][1] + '.csv')