import json

def extract_fields_from_json(input_json_file, output_json_file):   # 保留需要的部分
    extracted_data = []

    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 打开输入的JSON文件并读取内容

    for entry in data:
        filtered_entry = {   # 这边可以随需求做更改
            "content": entry["content"],
        }
        extracted_data.append(filtered_entry)  # 将提取的数据添加到列表
    # 提取每条记录中的需要的字段

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)   # 以JSON格式保存数据，缩进为4
    # 将提取的数据写入到新的JSON文件中

# 使用示例
input_json_file = 'data.json'
output_json_file = 'output.json'
extract_fields_from_json(input_json_file, output_json_file)

print('转换完成')
