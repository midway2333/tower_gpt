import sentencepiece as spm
import numpy as np
import os

# 加载SentencePiece模型
def load_sentencepiece_model(model_file):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)   # type: ignore
    return sp

# 编码文件夹中的所有txt文件并分别存储为二进制文件
def encode_and_store_individual_files(folder_path, model_file, output_folder):
    sp = load_sentencepiece_model(model_file)

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                encoded_ids = sp.encode_as_ids(text)   # type: ignore

            # 将编码ID转换为NumPy数组
            encoded_array = np.array(encoded_ids, dtype=np.uint32)

            # 保存为二进制文件
            output_file = os.path.join(output_folder, f'{os.path.splitext(file_name)[0]}.bin')
            encoded_array.tofile(output_file)

# 使用
folder_path = ''
model_file = ''
output_folder = ''

encode_and_store_individual_files(folder_path, model_file, output_folder)

print('保存完毕')
