import json

class RewriteJson:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def work(self):

        with open(self.input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        # 读取JSON文件

        print(data)
        # 打印数据以验证内容

        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
        # 将数据写入新的JSON文件

# 使用
rewriter = RewriteJson('sft_v.jsonl', 'v.jsonl')
rewriter.work()
