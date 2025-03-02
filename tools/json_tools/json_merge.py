import os
import json

def merge_json_files(input_folder, output_file):
    merged_data = []

    # 遍历给定文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as infile:
                try:
                    data = json.load(infile)
                    # 如果data不是列表，就把它转换成单元素列表
                    if not isinstance(data, list):
                        data = [data]
                    merged_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON from {filename}: {e}")
    
    # 写入合并后的数据到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    input_folder = 'data\\in'  # 替换为你的JSON文件所在的文件夹路径
    output_file = 'output.json'  # 替换为你想要保存合并后JSON文件的路径
    merge_json_files(input_folder, output_file)