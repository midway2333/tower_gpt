import json

def unescape_unicode_to_utf8(json_obj):
    """
    遍历 JSON 对象，将其所有的字符串值从 Unicode 转义序列转换为 UTF-8 字符串。
    """
    if isinstance(json_obj, dict):
        return {k: unescape_unicode_to_utf8(v) for k, v in json_obj.items()}
    elif isinstance(json_obj, list):
        return [unescape_unicode_to_utf8(element) for element in json_obj]
    elif isinstance(json_obj, str):
        # 将字符串中的 Unicode 转义序列转换为实际的 Unicode 字符
        return json_obj.encode('utf-8', 'surrogatepass').decode('utf-8')
    else:
        return json_obj

input_file_path = 'train_a.jsonl'
output_file_path = 'trainp.jsonl'

# 读取并处理每一行 JSON 数据
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:

    for line in infile:
        # 解析 JSON 行
        data = json.loads(line)
        # 转换其中的 Unicode 转义序列为 UTF-8 字符串
        converted_data = unescape_unicode_to_utf8(data)
        # 将转换后的数据写入新的文件
        json.dump(converted_data, outfile, ensure_ascii=False)
        outfile.write('\n')

print("转换完成")