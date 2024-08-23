import torch
from torch import nn
from tfer_chat import *
from tfer_dataloader import *
import sentencepiece as spm
from galore_torch import GaLoreAdamW8bit


sp = spm.SentencePieceProcessor()
sp.load('tokenizer\\spm_dict.model')   # type: ignore
vocab_size = sp.GetPieceSize()
# 构建词表

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判断引入设备

model = transformer(decoder_num=3, head_num=4, d=512, dk=256, dff=1024, vocab_size=vocab_size)
# 初始化模型

def train(epoch:int, folder_path: str, block_size:int, batch_size: int):
    loss_fn = nn.CrossEntropyLoss()   # 交叉熵loss

    model.train()   # 设置为训练模式
    optimizer = GaLoreAdamW8bit(model.parameters(), lr=rating, betas=(0.9, 0.999),   \
                                 eps=1e-8, weight_decay=0.01)

    dataloader = tfer_DataLoader(folder_path, block_size, batch_size, device)


    for i in range(1, epoch + 1):

        loss = 0

        for a in range(256):

            x, y = dataloader.get_batch()
            x, y = x.to(device).long(), y.to(device).long()   # 将数据移动到指定设备上
            pred = model(x)
            pred = pred.view(-1, pred.size(-1))
            # 将pred重新形状为 [batch_size * sequence_length, vocab_size]

            y = y.view(-1)
            # 将 y 重新形状为 [batch_size * sequence_length]

            loss = loss_fn(pred, y)   # 计算损失
            loss.backward()           # 反向传播,计算梯度
            optimizer.step()          # 更新模型参数
            optimizer.zero_grad()     # 清空梯度

            if (i+1) % 10 == 0 and a == 0:
                if isinstance(loss, torch.Tensor):
                    print(f'Epoch: {i+1}; Loss: {loss.item()}; block_size: {block_size}')   # type:ignore
                else:
                    print(f'Epoch: {i+1}; Loss: {loss}; block_size: {block_size}')   # type:ignore

            # 每10个epoch打印一次当前的epoch,损失,block_size


block_size = 512
rating = 0.0003

folder_path = ''
train(epoch=300, block_size=block_size, folder_path=folder_path, batch_size=4)

torch.save(model.state_dict(), '')