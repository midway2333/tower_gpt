import os

def merge_jsonl_files_streaming(input_directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_directory):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(input_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)

input_dir = 'data\\t'  # 替换为你的输入文件夹路径
output_file = 'train.jsonl'  # 替换为你想要保存的输出文件路径

# 使用流式处理函数
merge_jsonl_files_streaming(input_dir, output_file)