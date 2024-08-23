import os

def batch_modify_txt_files(folder_path, old_content, new_content):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                file_data = file.read()
            
            # 替换旧内容为新内容
            file_data = file_data.replace(old_content, new_content)
            
            # 将修改后的内容写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(file_data)
                
    print("所有txt文件内容已修改完成!")

# 使用

folder_path = ''
old_content = ''
new_content = ''
batch_modify_txt_files(folder_path, old_content, new_content)
