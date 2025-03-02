import json

def combine(input_json_file, output_json_file):
    combined_data = []

    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 打开输入的JSON文件并读取内容

    for entry in data:
        combined_entry = {   # 不同的就改这里
            "combined_input": entry["instruction"] + entry["input"],
            "output": entry["output"]
        }
        combined_data.append(combined_entry)  # 将拼接后的数据添加到列表
    # 拼接

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)  # 以JSON格式保存数据，缩进为4
    # 将拼接后的数据写入到新的JSON文件中

# 使用
input_json_file = 'alpaca_gpt4_data_zh.json'
output_json_file = 'dpo.json'
combine(input_json_file, output_json_file)

print('拼接完成')
