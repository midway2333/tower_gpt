import json

# 打开JSONL文件
with open('data\\part-663de978334d-000000.jsonl', 'r', encoding='utf-8') as jsonl_file:
    # 打开TXT文件以写入
    with open('news.txt', 'w', encoding='utf-8') as txt_file:
        # 逐行读取JSONL文件
        for line in jsonl_file:
            # 将每行解析为JSON对象
            json_obj = json.loads(line)
            # 提取正文部分
            content = json_obj.get('content', '')
            # 将正则表达式替换为可读形式
            readable_content = content.replace('\\n', '\n')
            # 写入TXT文件
            txt_file.write(readable_content + '\n')

print("转换完成!")