import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sentencepiece as spm
from galore_torch import GaLoreAdamW8bit
from torch.utils.tensorboard import SummaryWriter   # type: ignore
from tfer_chat import *

class SequentialDataset(Dataset):   # 按顺序遍历文件
    def __init__(self, data_dir, block_size, padding_id):

        """

        初始化SequentialDataset类的实例

        参数:
        - data_dir (str): 数据文件所在的目录
        - block_size (int): 单个输入中的文本长度
        - padding_id (int): 填充标识符的ID

        """

        self.data_dir = data_dir
        self.block_size = block_size
        self.padding_id = padding_id

        self.files = [f for f in os.listdir(self.data_dir)   \
            if os.path.isfile(os.path.join(self.data_dir, f)) and f.endswith('.bin')]
        # 获取文件夹中的所有.bin文件
        
    def __len__(self):   # 返回数据集中文件的数量
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.memmap(os.path.join(self.data_dir, file), dtype=np.uint32, mode='r')

        if len(data) > self.block_size:

            i = random.randint(0, len(data) - self.block_size -1)
            # 随机选择一个起始索引

            x_data = data[i:i+self.block_size]
            y_data = data[i+1:i+1+self.block_size]

        else:
            x_data = np.pad(data[:len(data)-1], (0, self.block_size - len(data) + 1),   \
                             'constant', constant_values=self.padding_id)
            y_data = np.pad(data[1:len(data)], (0, self.block_size - len(data) + 1),   \
                             'constant', constant_values=self.padding_id)

        x = torch.from_numpy(x_data.astype(np.int64))
        y = torch.from_numpy(y_data.astype(np.int64))

        return x, y
    

def train(epoch: int, dataloader: DataLoader, model: nn.Module,   \
           device: torch.device, writer, rating: float, wr_name):

    """

    训练模型的函数

    参数:
    - epoch (int): 训练的轮数
    - dataloader (DataLoader): 数据加载器
    - model (nn.Module): 要训练的模型
    - device (torch.device): 设备类型(CPU或GPU)
    - writer: 用于记录训练日志的对象
    - rating (float): 学习率

    """
    
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

    model.train()  # 设置为训练模式
    optimizer = GaLoreAdamW8bit(model.parameters(), lr=rating,   \
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    train_steps = 0  # 初始化训练步数

    for i in range(epoch):

        epoch_loss = 0

        for x, y in dataloader:

            x, y = x.to(device).long(), y.to(device).long()  # 将数据移动到指定设备上
            pred = model(x)
            pred = pred.view(-1, pred.size(-1))
            # 将 pred 重新形状为 [batch_size * sequence_length, vocab_size]

            y = y.view(-1)
            # 将 y 重新形状为 [batch_size * sequence_length]

            loss = loss_fn(pred, y)     # 计算损失
            loss.backward()             # 反向传播，计算梯度

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 梯度裁剪

            optimizer.step()            # 更新模型参数
            optimizer.zero_grad()       # 清空梯度

            epoch_loss += loss.item()   # 累加batch损失
            avg_epoch_loss = epoch_loss / len(dataloader)   # 计算平均epoch损失

        if (i + 1) % 10 == 0:
            if isinstance(loss, torch.Tensor):
                print(f'Epoch: {i + 1}; avg_Loss: {avg_epoch_loss}')
            else:
                print(f'Epoch: {i + 1}; avg_Loss: {avg_epoch_loss}')
        # 打印当前的epoch与loss

        train_steps += 1
        writer.add_scalar(wr_name, avg_epoch_loss, train_steps)  # 记录训练损失
    
if __name__ == "__main__":

    data_dir = 'nptxt'
    block_size = 512
    batch_size = 4
    # dataset设置

    sp = spm.SentencePieceProcessor()
    sp.load('work\\tokenizer\\spm_dict.model')   # type: ignore
    vocab_size = sp.GetPieceSize()
    padding_id = sp.pad_id()
    # 构建词表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设备获取

    model = transformer(vocab_size=vocab_size, padding_idx=padding_id)
    # 模型设置

    epoch = 300
    rating = 0.0001
    writer = SummaryWriter('tr_logs')
    name = 'train_loss_0.0001'
    # 训练设置


    dataset = SequentialDataset(data_dir, block_size, padding_id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,   \
                                num_workers=4, pin_memory=True)
    

    train(epoch=epoch, dataloader=dataloader, model=model, \
            device=device, writer=writer, rating=rating, wr_name=name)

    torch.save(model.state_dict(), 'test7m_0.001.pth')
