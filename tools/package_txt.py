import os

def combine_txt_files(folder_path, output_file):
    combined_text = ""
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as file:
            combined_text += file.read() + "\n"
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(combined_text)
    
    print(f"所有txt文件内容已合并到 {output_file} 中!")

# 使用
folder_path = ''
output_file = ''
combine_txt_files(folder_path, output_file)
