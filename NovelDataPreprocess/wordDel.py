import os

# 讀取資料
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 建立文字對照表的字典
def create_char_frequency_dict(text, char_set):
    frequency_dict = {char: 0 for char in char_set}
    for char in text:
        if char in frequency_dict:
            frequency_dict[char] += 1
    return frequency_dict

# 載入字對照表
def load_char_set(char_set_file):
    with open(char_set_file, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

# 保存字表到文件
def save_filtered_char_list(char_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for char in char_list:
            file.write(char + '\n')

# 處理多個文件
def process_multiple_files(file_paths, char_set, output_file):
    char_frequency_dict = {char: 0 for char in char_set}
    
    for file_path in file_paths:
        text = load_text_file(file_path)
        for char in text:
            if char in char_frequency_dict:
                char_frequency_dict[char] += 1
    
    # 選出出現次數大裕等於2的字
    filtered_char_list = [char for char, frequency in char_frequency_dict.items() if frequency >= 3]
    
    # 把選出來的字出存成字對照表
    save_filtered_char_list(filtered_char_list, output_file)

folder_path = 'NovelTxt'
file_paths_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.relpath(os.path.join(root, file), folder_path)
        file_paths_list.append(folder_path + '\\' + file_path)

# 主程式
def main():
    char_set = load_char_set('vocab\\word_embedding_vocab.txt')
    output_file = 'vocab\\wordDel.txt'
    
    process_multiple_files(file_paths_list, char_set, output_file)

if __name__ == '__main__':
    main()