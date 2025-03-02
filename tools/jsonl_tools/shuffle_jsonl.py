import json
import random

def shuffle_jsonl(input_file, output_file):
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 打乱所有行
    random.shuffle(lines)
    
    # 写入打乱后的行到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            # 去除行尾的换行符后写入
            f.write(line)

# 使用函数
input_file = '111.jsonl'  # 输入的JSONL文件路径
output_file = 'train.jsonl'  # 输出的打乱后的JSONL文件路径
shuffle_jsonl(input_file, output_file)