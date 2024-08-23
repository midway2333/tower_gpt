import base64

def decode_base64_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                # 去除行末的换行符
                line = line.strip()
                # 将编码部分和数字部分分开
                encoded_str, number = line.split()
                # 解码 Base64 字符串
                decoded_str = base64.b64decode(encoded_str).decode('ascii', errors='ignore')
                # 将解码后的字符串和数字重新组合并写入输出文件
                outfile.write(f"{decoded_str} {number}\n")
            except Exception as e:
                # 如果有错误，将错误信息写入输出文件
                outfile.write(f"Error decoding: {line}\n")

# 输入文件和输出文件的路径
input_file = ""
output_file = ""

# 解码并写入新文件
decode_base64_file(input_file, output_file)

print(f"文件已处理完成，结果保存在 {output_file}")
