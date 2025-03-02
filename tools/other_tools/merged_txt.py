import os
import glob

"""
合并txt文件
"""

def merge_txt_files(folder_path, output_filename='merged_output.txt'):
    # 检查给定路径是否存在且为目录
    if not os.path.isdir(folder_path):
        print(f"The provided path {folder_path} is not a valid directory.")
        return
    
    # 获取文件夹中所有的txt文件
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # 如果没有找到任何txt文件，则退出
    if not txt_files:
        print("No .txt files found in the specified directory.")
        return
    
    # 打开输出文件准备写入
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as infile:
                # 读取文件内容并写入到输出文件中
                outfile.write(f"--- Begin content from {os.path.basename(txt_file)} ---\n")
                outfile.write(infile.read())
                outfile.write("\n--- End content from {os.path.basename(txt_file)} ---\n\n")
                
    print(f"All .txt files have been merged into {output_filename}")

# 使用示例
folder_to_merge = 'data\\mgz'  # 替换为你的文件夹路径
merge_txt_files(folder_to_merge)