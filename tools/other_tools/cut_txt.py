import os

def split_file_by_size(file_path, chunk_size_mb=100, output_dir=None):

    """
    
    - file_path (str): 要分割的源文件路径
    - output_dir (str): 输出文件存放的目录,默认为None,表示与源文件同一目录

    """

    # 将MB转换为字节
    chunk_size = chunk_size_mb * 1024 * 1024 / 3
    
    # 获取文件名和扩展名
    file_name, ext = os.path.splitext(os.path.basename(file_path))
    
    # 如果提供了输出目录，则确保它存在
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 初始化输出文件编号、当前分块大小
    output_chunk_number = 1
    current_chunk_size = 0
    
    # 打开源文件进行读取
    with open(file_path, 'r', encoding='utf-8') as infile:
        # 确保首次迭代时有输出文件打开
        output_file_name = f'{file_name}_part_{output_chunk_number}{ext}'
        if output_dir:
            output_file_name = os.path.join(output_dir, output_file_name)
        output_file = open(output_file_name, 'w', encoding='utf-8')
        
        # 遍历源文件的每一行
        for line in infile:
            # 如果添加下一行后会超过设定的分块大小，则开启新的分块文件
            if current_chunk_size + len(line) >= chunk_size:
                # 关闭当前输出文件
                output_file.close()
                
                # 增加分块文件编号
                output_chunk_number += 1
                
                # 创建新的分块文件名，并打开用于写入
                output_file_name = f'{file_name}_part_{output_chunk_number}{ext}'
                if output_dir:
                    output_file_name = os.path.join(output_dir, output_file_name)
                output_file = open(output_file_name, 'w', encoding='utf-8')
                
                # 重置当前分块大小
                current_chunk_size = 0
            
            # 写入当前行到输出文件中
            output_file.write(line)
            
            # 更新当前分块大小
            current_chunk_size += len(line)

        # 确保最后的输出文件被关闭
        if not output_file.closed:
            output_file.close()

# 使用
split_file_by_size('data\\mgz\\综合1.txt', 300, output_dir='data\\cut')
print('转换完成')
