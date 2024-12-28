from re import L
from torch.utils.data import Dataset, IterableDataset
from numpy.random import shuffle
import random, torch, json
import sentencepiece as spm


class DialogueDataProcessor:
    def __init__(self, json_file, sp_model_path, block_size):

        """

        初始化DialogueDataProcessor类的实例

        参数:
        - json_file (str): 包含对话数据的JSON文件路径
        - sp_model_path (str): SentencePiece模型文件路径
        - block_size (int): 单个输入中的最大token数量

        """

        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)   # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int=8192   # 缓冲区大小
        self.length = 0   # 初始化用于生成器下记录数据集长度
    
    def load_and_encode_data(self):

        """

        加载并编码对话数据
        使用json文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表

        """

        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 打开并读取JSON文件

        inputs = []
        targets = []

        for dialogue in data:   # 遍历每个文本
            input = dialogue['lines']   # <<<对于不同的训练集可能需要在此修改

            input_ids = [self.bos_id] + self.sp.encode(input, out_type=int) + [self.eos_id]   # type: ignore
            response_ids = [self.bos_id] + self.sp.encode(input, out_type=int) + [self.eos_id]    # type: ignore
            # 使用SentencePiece分词器进行编码

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            response_ids = torch.tensor(response_ids, dtype=torch.long)
            # 将编码信息转换为tensor


            if len(input_ids) > self.block_size:   # 随机选择一个起始索引

                i = random.randint(0, len(input_ids) - self.block_size -1)
                x_data = input_ids[i:i+self.block_size]
                y_data = response_ids[i+1:i+1+self.block_size]

                inputs.append(x_data)
                targets.append(y_data)

            else:   # 如果文件长度小于block_size,舍弃
                pass
        
        return inputs, targets


    def data_generator(self):

        """

        使用生成器加载并编码对话数据,适用于大数据集加载
        使用jsonl文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表

        """

        self.length = 0   # 重置长度
        with open(self.json_file, 'r', encoding='utf-8') as f:
            buffer_list = []   # 缓冲区

            for line in f:   # 加载jsonl文件,或者说非数组json
                dialogue = json.loads(line.strip())

                input = dialogue['lines']   # 获取用户输入文本
                input_ids = self.sp.encode(input, out_type=int)   # type: ignore

                input_ids = torch.tensor(input_ids, dtype=torch.long)
                # 将编码信息转换为tensor

                if len(input_ids) > self.block_size:   # 随机选择一个起始索引
                    self.length += 1   # 记录长度

                    i = random.randint(0, len(input_ids) - self.block_size -1)
                    x_data = input_ids[i:i+self.block_size]
                    y_data = input_ids[i+1:i+1+self.block_size]

                    if len(buffer_list) < self.buffer_size:   # 添加数据
                        buffer_list.append( (x_data, y_data) )
                        continue

                    shuffle(buffer_list)   # 缓存区满了,返回数据
                    for inputs, targets in buffer_list:
                        
                        yield inputs, targets   # 迭代

                else:   # 舍弃
                    pass
            
            if buffer_list:   # 处理剩余的缓冲区数据
                shuffle(buffer_list)
                for inputs, targets in buffer_list:
                    yield inputs, targets  # 迭代

    def __len__(self) -> int:   # 返回长度
        return self.length


class DialogueDataset(Dataset):   # 负责加载和编码数据的实例
    def __init__(self, processor):
        self.inputs, self.targets = processor.load_and_encode_data()
    
    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)
    
    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx]
    
class GeneratorDialogueDataset(IterableDataset):   # 负责生成器模式下加载和编码数据的实例
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
