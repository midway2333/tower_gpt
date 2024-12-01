import os
import json

def txt_to_json(input_file_path, output_file_path, lines_per_json=100):

    """

    将单个文本文件转换为JSON文件
    
    - param input_file_path: 输入的文本文件路径
    - param output_file_path: 输出的JSON文件路径
    - param lines_per_json: 每个JSON对象包含的行数,默认为100

    """

    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 打开并读取输入文件的所有行    

    json_objects = []
    # 创建一个空列表来存储JSON对象    

    for i in range(0, len(lines), lines_per_json):   # 创建JSON对象
        chunk = lines[i:i + lines_per_json]
        # 获取当前组的行

        chunk = [line.strip() for line in chunk]
        # 去除每行末尾的换行符

        combined_line = ''.join(line.strip() for line in chunk)
        # 去除每行末尾的换行符并拼接成一个字符串

        json_object = {"lines": combined_line}
        # 创建一个JSON对象

        json_objects.append(json.dumps(json_object, ensure_ascii=False))
        # 将JSON对象添加到列表中

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for json_obj in json_objects:
            file.write(json_obj + '\n')
    # 将所有的JSON对象写入到输出文件中

def process_folder(folder_path, output_path, lines_per_json=100):

    """

    处理指定文件夹中的所有文本文件,并将它们转换为JSON文件
    
    - param folder_path: 包含文本文件的文件夹路径
    - param lines_per_json: 每个JSON对象包含的行数,默认为100

    """


    for filename in os.listdir(folder_path):   # 遍历文件夹中的所有文件

        if filename.endswith('.txt'):   # 检查文件是否为文本文件
            input_file_path = os.path.join(folder_path, filename)   # 构建输入文件的完整路径
            output_file_path = os.path.join(output_path, filename.replace('.txt', '.json'))
            # 构建输出文件的完整路径,将扩展名从txt改为json

            print(f"Processing {filename}...")   # 打印正在处理的文件名
            txt_to_json(input_file_path, output_file_path, lines_per_json)
            # 调用 txt_to_json 函数进行转换
            print(f"Converted {filename} to {os.path.basename(output_file_path)}")
            # 打印转换完成的信息

# 使用
folder_path = 'data\\ChineseBook'
output_path = 'data\\bookjson'

process_folder(folder_path, output_path)
print('已完成')
