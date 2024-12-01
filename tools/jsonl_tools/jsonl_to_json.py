import json

def jsonl_to_json(jsonl_file, json_file):
    data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    with open(json_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False, indent=4)
            f.write('\n')  # 写入每条记录后换行

# 使用
jsonl_file = 'data\\bookjson\\文学作品.json'
json_file = 'booktxt.json'
jsonl_to_json(jsonl_file, json_file)

print('转换完成')
