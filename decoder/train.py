import json
import sentencepiece as spm
import torch
from galore_torch import GaLoreAdamW8bit
from torch.utils.tensorboard import SummaryWriter   # type: ignore
from decoder.model import *
from torch import GradScaler, autocast   # type: ignore
from torch.utils.data import Dataset, DataLoader
import random
from datetime import datetime
from typing import Optional

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
        self.bos_id = sp.bos_id()
        self.eos_id = sp.eos_id()
    
    def load_and_encode_data(self):

        """

        加载并编码对话数据

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

            input_ids = self.sp.encode(input, out_type=int)   # type: ignore
            response_ids = self.sp.encode(input, out_type=int)   # type: ignore

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
    
class DialogueDataset(Dataset):   # 负责加载和编码数据的实例
    def __init__(self, processor):
        self.inputs, self.targets = processor.load_and_encode_data()
    
    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)
    
    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx]

def train(
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    writer,
    rating: float,
    wr_name,
    steps: int,
    train_steps: int = 0,
    test_set: bool = False,
    test_dataloader: Optional[DataLoader] = None
):

    """

    训练模型的函数

    参数:
    - epoch (int): 训练的轮数
    - dataloader (DataLoader): 数据加载器
    - model (nn.Module): 要训练的模型
    - device (torch.device): 设备类型(CPU或GPU)
    - writer: 用于记录训练日志的对象
    - rating (float): 学习率
    - wr_name: 训练日志名称
    - steps: 梯度累积步进
    - train_steps: 断点续训轮数
    - test_set: 是否运用测试集
    - test_dataloader: 测试数据加载器

    """

    loss_fn = nn.CrossEntropyLoss()   # 交叉熵损失函数
    accumulation_steps = steps   # 设置累积步数
    scaler = GradScaler()
    optimizer = GaLoreAdamW8bit(model.parameters(), lr=rating,   \
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    # 优化器

    train_steps = train_steps   # 初始化训练步数

    for i in range(epoch):
        model.train()   # 设置为训练模式
        epoch_loss = 0
        train_steps += 1
        optimizer.zero_grad()  # 清除之前的梯度

        for step, (x, y) in enumerate(dataloader):   # 生成步进索引

            x, y = x.to(device).long(), y.to(device).long()   # 将数据移动到指定设备上

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # 自动混合精度

                pred = model(x)
                pred = pred.view(-1, pred.size(-1))
                # 将 pred 重新形状为 [batch_size * sequence_length, vocab_size]

                y = y.view(-1)
                # 将 y 重新形状为 [batch_size * sequence_length]

                loss = loss_fn(pred, y) / accumulation_steps   # 计算损失

            scaler.scale(loss).backward()
            # 使用混合精度来缩放损失,反向传播
            
            if (step + 1) % accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 梯度裁剪

                scaler.step(optimizer)   # 更新模型参数
                scaler.update()          # 调整缩放比例

            epoch_loss += loss.item()    # 累加batch损失

        avg_epoch_loss = (epoch_loss / len(dataloader)) * accumulation_steps
        # 计算平均epoch损失

        if test_set and test_dataloader is not None:

            model.eval()   # 设置为评估模式
            epoch_test_loss = 0

            with torch.no_grad():  # 不需要计算梯度
                for tx, ty in test_dataloader:
                    tx, ty = tx.to(device).long(), ty.to(device).long()

                    test_pred = model(tx)
                    test_pred = test_pred.view(-1, test_pred.size(-1))
                    ty = ty.view(-1)
                    # 同上

                    test_loss = loss_fn(test_pred, ty)
                    epoch_test_loss += test_loss.item()
                
            avg_test_loss = (epoch_test_loss / len(test_dataloader))
            # 计算损失

            if (i + 1) % 10 == 0:   # 拥有测试集时打印对应loss
                print(f'Epoch: {i + 1}; avg_Loss: {avg_epoch_loss}; avg_test_Loss: {avg_test_loss}')
            writer.add_scalar(wr_name+'_test', avg_test_loss, train_steps)   # 记录测试损失


        else:
            if (i + 1) % 10 == 0:   # 没有测试集时打印对应loss
                print(f'Epoch: {i + 1}; avg_Loss: {avg_epoch_loss}')

        writer.add_scalar(wr_name, avg_epoch_loss, train_steps)   # 记录训练损失

def train_log(model_name, log_file, block_size, batch_size, epoch,   \
                       rating, step, writer, name):

    """

    - model_name: 模型名
    - log_file: 记录文件的路径
    - block_size: 窗口大小
    - batch_size: batch大小
    - epoch: 训练轮次
    - rating: 学习率
    - step: 梯度累计步进
    - writer: tensorboard文件夹
    - name: tensorboard_log名称

    """

    # 创建一个新的记录条目
    log_entry = {
        "time": datetime.now().isoformat(),
        "model": model_name,
        "block_size": block_size,
        "batch_size": batch_size,
        "epoch": epoch,
        "learning_rate": rating,
        "step": step,
        "writer": writer,
        "name": name
    }

    logs = []   # 读取现有的日志文件内容
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # 如果文件不存在或为空,则创建一个空列表

    if logs:
        last_entry = logs[-1]
        previous_epoch_sum = last_entry.get("epoch", 0)
        log_entry["epoch"] = last_entry.get("epoch", 0) + epoch
    else:
        log_entry["epoch"] = epoch
        previous_epoch_sum = 0
    # 获取最新的记录以累加epoch

    logs.append(log_entry)
    # 追加新的记录

    with open(log_file, 'w') as file:
        json.dump(logs, file, indent=4)
    # 将更新后的日志列表写回文件

    return previous_epoch_sum
    # 返回当前轮次开始时的epoch轮次
    # 用于记录上一次断点


if __name__ == '__main__':

    file_path = 'data\\bookdata.json'
    block_size = 128
    batch_size = 12
    # 训练集设置

    test_file_path = 'data\\test.json'
    test_block_size = 128
    test_batch_size = 12
    # 测试集设置

    sp = spm.SentencePieceProcessor()
    sp_path = 'tokenizer\\spm_dict_v2.1.model'
    sp.load(sp_path)   # type: ignore
    vocab_size = sp.GetPieceSize()
    padding_id = sp.pad_id()
    # 构建词表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设备获取

    model = transformer(vocab_size=vocab_size, padding_idx=padding_id)
    # 模型设置

    epoch = 30
    rating = 0.0001
    step = 32
    writer_file = 'tr_logs'
    writer = SummaryWriter(writer_file)
    name = 'spm_dict_v2.1'
    # 训练设置

    model_name = 'spm_dict_v2.1.bin'
    log_file = 'spm_dict_v2.1.log'
    ep = train_log(model_name, log_file, block_size, batch_size, epoch,   \
                       rating, step, writer_file, name)
    # 信息设置

    if ep != 0:
        model.load_state_dict(torch.load(model_name, weights_only=True))
    # 断点续训

    data_processor = DialogueDataProcessor(file_path, sp_path, block_size)
    dataset = DialogueDataset(data_processor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,   \
                                num_workers=4, pin_memory=True)
    # 训练集 dataset @ dataloader

    test_data_processor = DialogueDataProcessor(test_file_path, sp_path, test_block_size)
    test_dataset = DialogueDataset(test_data_processor)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,   \
                                num_workers=4, pin_memory=True)
    # 训练集 dataset @ dataloader
    
    train(epoch=epoch,
        dataloader=dataloader,
        model=model,
        device=device,
        writer=writer,
        rating=rating,
        wr_name=name,
        steps=step,
        train_steps=ep,
        test_set=True,
        test_dataloader=test_dataloader
        )

    torch.save(model.state_dict(), model_name)
