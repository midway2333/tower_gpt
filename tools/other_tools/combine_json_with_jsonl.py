import json

# 读取原始的JSON文件
with open('gpt4_data.json', 'r', encoding='utf-8') as original_file:
    original_data = json.load(original_file)

# 读取jsonl文件
jsonl_data = []
with open('output_results.json', 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        jsonl_data.append(json.loads(line.strip()))

# 合并数据
if len(original_data) != len(jsonl_data):
    raise ValueError("The number of lines in the JSONL file does not match the number of items in the original JSON file.")

for i in range(len(original_data)):
    # 将'reject'字段添加到每个item中
    reject_value = jsonl_data[i].get('reject')
    if reject_value:  # 如果'reject'存在且不为空
        original_data[i]['reject'] = reject_value

# 将合并后的数据写入新的JSON文件
with open('dpo3.json', 'w', encoding='utf-8') as output_file:
    json.dump(original_data, output_file, ensure_ascii=False, indent=2)

print("数据合并完成")