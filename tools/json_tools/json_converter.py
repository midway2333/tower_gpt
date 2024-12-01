import json

class JSONConverter:
    def __init__(self, input_file, output_file, encoding='utf-8'):

        """

        初始化JSONConverter类的实例

        参数:
        - input_file (str): 原始JSON文件路径
        - output_file (str): 转换后的JSON文件路径
        - encoding (str): 文件编码，默认为'utf-8'

        提示:
        - 如果json文件格式为:
        - {data}
        - {data}
        - {data}
        - 
        - 希望加上[]让它可以送入训练
        - 就用这个文件

        """

        self.input_file = input_file
        self.output_file = output_file
        self.encoding = encoding
    
    def convert(self):
    # 将多个独立的JSON对象转换为一个数组,并写入新的JSON文件


        with open(self.input_file, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        # 读取原始JSON文件

        json_objects = [json.loads(line) for line in lines]
        # 将每一行的JSON对象转换为一个数组

        with open(self.output_file, 'w', encoding=self.encoding) as f:
            json.dump(json_objects, f, ensure_ascii=False, indent=4)
        # 将数组写入新的JSON文件


# 使用
input_file = 'data\\bookjson\\文学作品.json'
output_file = 'bookdata.json'

converter = JSONConverter(input_file, output_file)
converter.convert()
