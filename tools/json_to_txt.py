import json
import os

with open('', 'r', encoding='utf-8') as file:
    data = json.load(file)
# 读取JSON数据

os.makedirs('', exist_ok=True)
# 创建存储文本文件的目录

output_filename = ''
with open(output_filename, 'w', encoding='utf-8') as text_file:
    for i, entry in enumerate(data):
        text_file.write(f"Entry {i+1}:\n")
        text_file.write(entry['completion'] + "\n\n")
# 遍历JSON数据并将所有条目保存为一个文本文件

print("所有条目已保存为一个文本文件")

