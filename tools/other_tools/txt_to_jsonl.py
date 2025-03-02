import os
import json

def txt_to_jsonl(input_folder, output_folder, max_chars_per_line=1000):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename.replace('.txt', '.jsonl'))

            # 打开输入文件和输出文件
            with open(input_file_path, 'r', encoding='utf-8') as txt_file, \
                 open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
                buffer = ""
                for line in txt_file:
                    buffer += line
                    # 如果缓冲区的字符数超过最大限制，写入JSONL文件并清空缓冲区
                    if len(buffer) >= max_chars_per_line:
                        jsonl_file.write(json.dumps({"text": buffer.strip()}, ensure_ascii=False) + '\n')
                        buffer = ""
                # 写入剩余的缓冲区内容
                if buffer:
                    jsonl_file.write(json.dumps({"text": buffer.strip()}, ensure_ascii=False) + '\n')

# 使用示例
input_folder = 'data\\cut'  # 输入文件夹路径
output_folder = 'data\\train'  # 输出文件夹路径
max_chars_per_line = 512  # 每行最大字符数

txt_to_jsonl(input_folder, output_folder, max_chars_per_line)
print('转换完成！')
