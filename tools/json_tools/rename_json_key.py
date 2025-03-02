import json

# 读取合并后的JSON文件
with open('111.json', 'r', encoding='utf-8') as merged_file:
    data = json.load(merged_file)

# 遍历每个条目并重命名字段
for item in data:
    # 保存原始值
    combined_input_value = item.pop('combined_input', None)
    output_value = item.pop('output', None)
    r1 = item.pop('rejected', None)
    r2 = item.pop('reject', None)

    # 重新赋值给新的字段名
    if combined_input_value is not None:
        item['prompt'] = combined_input_value
    if output_value is not None:
        item['chosen'] = output_value
    if r1 is not None:
        item['reject'] = r1
    if r2 is not None:
        item['reject'] = r2

# 将修改后的数据写回JSON文件
with open('alpaca_gpt4_data_zh.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)

print("字段重命名完成")