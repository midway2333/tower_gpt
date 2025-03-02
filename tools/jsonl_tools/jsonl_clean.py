import os
import string
import json

# 定义要处理的标点符号列表（包括中文和英文）
punctuation_list = string.punctuation + '!#$%&()*+,-.．/:;<=>?@[]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|:："'

def clean_text(text, space_to_comma=True):
    """
    清洗文本内容：将空格转换为逗号，并确保连续的标点符号最多只出现一次。
    
    :param text: 要清洗的文本
    :param space_to_comma: 是否将空格转换为逗号，默认为 True
    :return: 清洗后的文本
    """
    cleaned_text = text
    
    if space_to_comma:
        # 将所有空格替换为逗号
        cleaned_text = cleaned_text.replace(' ', ',')
        cleaned_text = cleaned_text.replace('　', ',')

    # 移除多余的连续标点符号
    for punct in punctuation_list:
        while punct * 2 in cleaned_text:
            cleaned_text = cleaned_text.replace(punct * 2, punct)

    return cleaned_text

def clean_json_object(json_obj, space_to_comma=True):
    """
    清洗 JSON 对象中的字符串值。
    
    :param json_obj: 要清洗的 JSON 对象
    :param space_to_comma: 是否将空格转换为逗号，默认为 True
    :return: 清洗后的 JSON 对象
    """
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, str):
                json_obj[key] = clean_text(value, space_to_comma)
            elif isinstance(value, (dict, list)):
                clean_json_object(value, space_to_comma)
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            if isinstance(item, str):
                json_obj[i] = clean_text(item, space_to_comma)
            elif isinstance(item, (dict, list)):
                clean_json_object(item, space_to_comma)
    return json_obj

def is_line_valid(cleaned_line, min_chars=10, max_space_punct_ratio=0.5):
    """
    检查清洗后的行是否有效：字符数不少于 min_chars，
    并且空格和标点符号的比例不超过 max_space_punct_ratio。
    
    :param cleaned_line: 清洗后的行
    :param min_chars: 行字符数的最小阈值，默认为 10
    :param max_space_punct_ratio: 空格和标点符号的最大比例，默认为 0.5
    :return: 如果行有效则返回 True，否则返回 False
    """
    stripped_line = cleaned_line.rstrip('\n')
    if len(stripped_line) < min_chars:
        return False
    
    # 计算非空格和非标点符号的数量
    valid_char_count = sum(1 for char in stripped_line if not (char == ',' or char in punctuation_list))
    
    # 计算非空格和非标点符号的比例
    valid_char_ratio = (valid_char_count / len(stripped_line)) if len(stripped_line) > 0 else 0
    
    return valid_char_ratio >= (1 - max_space_punct_ratio)

def filter_jsonl_files(directory, min_chars=10, max_space_punct_ratio=0.5):
    """
    遍历给定目录下的所有 .jsonl 文件，在原文件上移除其中字符数（不包括换行符）少于 min_chars 的行，
    或者空格和标点符号比例大于 max_space_punct_ratio 的行，并执行行内容清洗。
    
    :param directory: 包含要处理的 .jsonl 文件的目录路径
    :param min_chars: 行字符数的最小阈值，默认为 10
    :param max_space_punct_ratio: 空格和标点符号的最大比例，默认为 0.5
    """
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(directory, filename)
            
            # Read all lines first to avoid file corruption during write operations
            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            # Process and write back to the same file
            with open(file_path, 'w', encoding='utf-8') as outfile:
                for line in lines:
                    try:
                        json_obj = json.loads(line)
                        cleaned_json_obj = clean_json_object(json_obj)
                        cleaned_line = json.dumps(cleaned_json_obj, ensure_ascii=False) + '\n'
                        if is_line_valid(cleaned_line, min_chars, max_space_punct_ratio):
                            outfile.write(cleaned_line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line}")

# 使用函数，设定最少字符数为20，空格和标点符号的最大比例为0.3
filter_jsonl_files('data\\train', min_chars=512, max_space_punct_ratio=0.20)