import json

# 输入和输出文件名
input_filename = 'clean_dpo_train_len.json'
output_filename = 'dpo.jsonl'

# 读取输入的JSON文件
with open(input_filename, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 写入到JSONL文件
with open(output_filename, 'w', encoding='utf-8') as outfile:
    for item in data:
        json_line = json.dumps(item, ensure_ascii=False)
        outfile.write(json_line + '\n')

print(f"转换完成，已保存至 {output_filename}")