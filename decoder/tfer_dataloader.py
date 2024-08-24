import os
import numpy as np
import torch
import random
import sentencepiece as spm

# 参考: https://github.com/karpathy/nanoGPT

sp = spm.SentencePieceProcessor()
sp.load('work\\tokenizer\\spm_dict.model')   # type: ignore
padding_id = sp.pad_id()


class tfer_DataLoader:
    def __init__(self, data_dir, block_size, batch_size, device):

        """

        初始化 DataLoader 类的实例

        参数:
        - data_dir (str): 数据文件所在的目录
        - block_size (int): 单个输入中的文本长度
        - batch_size (int): 每个批次的大小
        - device (torch.device): 设备类型(CPU或GPU)

        """

        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device

    def get_batch(self):

        """

        从数据文件夹中随机选择batch_size个文件,读取长度为block_size的数据
        如果文件长度小于block_size,则选定整个文件

        将其转换为适合训练模型的格式

        返回:
        - x (torch.Tensor): 输入张量
        - y (torch.Tensor): 目标张量

        """

        files = [f for f in os.listdir(self.data_dir) if   \
                 os.path.isfile(os.path.join(self.data_dir, f))]
        # 获取文件夹中的所有文件

        selected_files = random.sample(files, self.batch_size)
        # 随机选择batch_size个文件

        x_list = []
        y_list = []
        
        for file in selected_files:

            data = np.memmap(os.path.join(self.data_dir, file), dtype=np.uint32, mode='r')

            if len(data) > self.block_size:   # 随机选择一个起始索引

                i = random.randint(0, len(data) - self.block_size -1)
                x_data = data[i:i+self.block_size]
                y_data = data[i+1:i+1+self.block_size]

            else:   # 如果文件长度小于block_size,则选定整个文件

                x_data = np.pad(data[:len(data)-1], (0, self.block_size - len(data) + 1),   \
                                 'constant', constant_values=padding_id)
                y_data = np.pad(data[1:len(data)], (0, self.block_size - len(data) + 1),   \
                                 'constant', constant_values=padding_id)
            
            x_list.append(torch.from_numpy(x_data.astype(np.int64)))
            y_list.append(torch.from_numpy(y_data.astype(np.int64)))

        x = torch.stack(x_list)
        y = torch.stack(y_list)
        # 将列表中的张量拼接成一个批次

        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        # 将数据移动到GPU(如果可用)

        return x, y
    
if __name__ == '__main__':

    device = 'cuda'
    dataloader = tfer_DataLoader('', 256, 50, device)

    for i in range(1,1001):
        x, y = dataloader.get_batch()

        print(x.shape)
        print(y.shape)
