import pandas as pd

class pd_to_jsonl:

    def __init__(self, path_pq, path_js):

        self.path_pq = path_pq
        self.path_js = path_js

    def work(self):

        df = pd.read_parquet(self.path_pq, engine='pyarrow')
        # 读取Parquet文件

        df.to_json(self.path_js, orient='records', lines=True)
        # 将DataFrame转换为JSONL文件

# 使用示例
converter = pd_to_jsonl('my_belll_1.5M_cn.parquet', 'train_a.jsonl')
converter.work()
