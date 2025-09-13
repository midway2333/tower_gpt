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

            input_ids = torch.tensor(input_ids, dtype=torch.int32)
            response_ids = torch.tensor(response_ids, dtype=torch.int32)
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

        with open(self.json_file, 'r', encoding='utf-8') as f:
            buffer_list = []   # 缓冲区

            for line in f:   # 加载jsonl文件,或者说非数组json
                dialogue = json.loads(line.strip())

                input = dialogue['lines']   # 获取用户输入文本
                input_ids = self.sp.encode(input, out_type=int)   # type: ignore

                input_ids = torch.tensor(input_ids, dtype=torch.int32)
                # 将编码信息转换为tensor

                if len(input_ids) > self.block_size:   # 随机选择一个起始索引

                    i = random.randint(0, len(input_ids) - self.block_size -1)
                    x_data = input_ids[i:i+self.block_size]
                    y_data = input_ids[i+1:i+1+self.block_size]

                    if len(buffer_list) < self.buffer_size:   # 添加数据
                        buffer_list.append( (x_data, y_data) )
                        continue

                    shuffle(buffer_list)   # 缓存区满了,返回数据
                    for inputs, targets in buffer_list:
                        yield inputs, targets   # 迭代

                    buffer_list = []   # 清空缓冲区

                else:   # 舍弃
                    pass
            
            if buffer_list:   # 处理剩余的缓冲区数据
                shuffle(buffer_list)
                for inputs, targets in buffer_list:
                    yield inputs, targets  # 迭代
    
    def data_length(self):
        """返回数据集的长度"""
        with open(self.json_file, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)


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


""" ------------------------------------- 以上为长文本dataset ------------------------------------- """

""" ------------------------------------- 以下对话文本dataset ------------------------------------- """

class Talk_DialogueDataProcessor:
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
        self.sp.load(sp_model_path)    # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int = 8192   # 添加缓冲区大小

        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        # 获得user_id与bot_id

    def load_and_encode_data(self):
        """加载并编码对话数据(用于小数据集)"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        inputs = []
        targets = []
        
        for dialogue in data:
            input_ids, response_ids = self._process_single_dialogue(dialogue)
            if input_ids is not None and response_ids is not None:
                inputs.append(input_ids)
                targets.append(response_ids)
        # 添加数据
        
        return inputs, targets

    def data_generator(self):
        """生成器方式加载并编码对话数据(用于大数据集)"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            buffer_list = []   # 缓冲区

            for line in f:   # 逐行读取文件内容
                dialogue = json.loads(line.strip())
                input_ids, response_ids = self._process_single_dialogue(dialogue)

                if len(buffer_list) < self.buffer_size:   # 将对话数据加入缓冲区
                    buffer_list.append((input_ids, response_ids))
                    continue   # 继续读取下一行

                shuffle(buffer_list)   # 缓存区满了,返回数据
                for inputs, targets in buffer_list:
                    yield inputs, targets

                buffer_list = []   # 清空缓冲区

            if buffer_list:   # 返回剩余数据
                shuffle(buffer_list)
                for inputs, targets in buffer_list:
                    yield inputs, targets

    def _process_single_dialogue(self, dialogue):
        """处理单个对话数据"""
        user_input = dialogue['prompt']   # <<<对于不同的训练集可能需要在此修改
        assistant_response = dialogue['response']   # <<<对于不同的训练集可能需要在此修改

        input_ids = [self.bos_id] + self.user_id + \
            self.sp.encode(user_input, out_type=int) + [self.eos_id]   # type: ignore
        response_ids = [self.bos_id] + self.bot_id + \
            self.sp.encode(assistant_response, out_type=int) + [self.eos_id]   # type: ignore

        input_all = input_ids + response_ids   # 总输入

        input_tensor = torch.tensor(input_all[:-1], dtype=torch.long)   # 去掉最后一个token用于创建目标
        target_tensor = torch.tensor(input_all[1:], dtype=torch.long)   # 移除第一个bos_token以匹配输入
        # 输入为对话的前n个token,目标为从第n+1个token开始到最后

        if len(input_tensor) > self.block_size:
            input_tensor = input_tensor[:self.block_size]
            target_tensor = target_tensor[:self.block_size]
        # 如果序列长度超过了block_size,截断

        padding_length = self.block_size - len(input_tensor)
        if padding_length > 0:
            input_tensor = torch.cat([input_tensor, torch.tensor([self.padding_id] * padding_length)])
            target_tensor = torch.cat([target_tensor, torch.tensor([self.padding_id] * padding_length)])
        # 填充到block_size

        return input_tensor, target_tensor

    def data_length(self):
        """返回数据集的总行数"""
        with open(self.json_file, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

class Talk_DialogueDataset(Dataset):
    """用于小数据集的数据加载器"""
    def __init__(self, processor):
        self.inputs, self.targets = processor.load_and_encode_data()
    
    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)
    
    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx]

class Talk_GeneratorDialogueDataset(IterableDataset):
    """用于大数据集的数据加载器"""
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
