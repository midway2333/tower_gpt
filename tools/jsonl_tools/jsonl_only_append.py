import json

def extract_fields_from_jsonl(input_jsonl_file, output_jsonl_file):
    with open(input_jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line.strip())
            
            # 提取需要的字段
            filtered_entry = {
                "lines": entry["content"]  # 根据需求修改字段
            }
            
            # 写入JSONL格式
            f_out.write(json.dumps(filtered_entry, ensure_ascii=False) + '\n')

# 使用示例
if __name__ == "__main__":
    extract_fields_from_jsonl("data\\part-663de978334d-000018.jsonl", "news2.jsonl")
    print('已完成!')