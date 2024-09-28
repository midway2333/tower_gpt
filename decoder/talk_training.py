import json
import sentencepiece as spm
import torch
from galore_torch import GaLoreAdamW8bit
from torch.utils.tensorboard import SummaryWriter   # type: ignore
from tfer_chat import *
from torch import GradScaler, autocast   # type: ignore
from torch.utils.data import Dataset, DataLoader


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

        for dialogue in data:   # 遍历每个对话
            user_input = dialogue['title']   # <<<对于不同的训练集可能需要在此修改
            assistant_response = dialogue['answer']   # <<<对于不同的训练集可能需要在此修改

            input_ids = [self.bos_id] +    \
                self.sp.encode(user_input, out_type=int) + [self.eos_id]   # type: ignore

            response_ids = [self.bos_id] +    \
                self.sp.encode(assistant_response, out_type=int) + [self.eos_id]   # type: ignore
            
            # 使用SentencePiece分词器进行编码

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            response_ids = torch.tensor(response_ids, dtype=torch.long)
            # 将编码信息转换为tensor

            if len(input_ids) + len(response_ids) > self.block_size:
                input_ids = input_ids[:self.block_size//2]
                response_ids = response_ids[:self.block_size//2]
            # 如果编码后的长度超过block_size,则截断文本

            input_ids = torch.cat([input_ids, torch.tensor([self.padding_id] *   \
                                    (self.block_size - len(input_ids)))])
            response_ids = torch.cat([response_ids, torch.tensor([self.padding_id] *   \
                                    (self.block_size - len(response_ids)))])
            # 填充input_ids和response_ids到block_size长度

            inputs.append(input_ids)
            targets.append(response_ids)
        
        return inputs, targets
    
class DialogueDataset(Dataset):
    def __init__(self, processor):
        self.inputs, self.targets = processor.load_and_encode_data()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train(epoch: int, dataloader: DataLoader, model: nn.Module,   \
           device: torch.device, writer, rating: float, wr_name, steps: int):

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

    """

    loss_fn = nn.CrossEntropyLoss()   # 交叉熵损失函数
    accumulation_steps = steps   # 设置累积步数
    scaler = GradScaler()

    model.train()   # 设置为训练模式
    optimizer = GaLoreAdamW8bit(model.parameters(), lr=rating,   \
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    train_steps = 0   # 初始化训练步数

    for i in range(epoch):

        epoch_loss = 0
        print('start step')

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

        if (i + 1) % 10 == 0:
            print(f'Epoch: {i + 1}; avg_Loss: {avg_epoch_loss}')
        # 打印当前的epoch与loss

        train_steps += 1
        writer.add_scalar(wr_name, avg_epoch_loss, train_steps)   # 记录训练损失

if __name__ == '__main__':

    file_path = 'train.json'
    sp_path = 'work\\tokenizer\\spm_dict.model'
    block_size = 256
    batch_size = 6
    # dataset/dataloader设置

    sp = spm.SentencePieceProcessor()
    sp.load('work\\tokenizer\\spm_dict.model')   # type: ignore
    vocab_size = sp.GetPieceSize()
    padding_id = sp.pad_id()
    # 构建词表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设备获取

    model = transformer(vocab_size=vocab_size, padding_idx=padding_id)
    model.load_state_dict(torch.load('test_tower_alpha', weights_only=True))
    # 模型设置

    epoch = 5
    rating = 0.0004
    step = 32
    writer = SummaryWriter('tr_logs')
    name = 'test_tower_talk'
    # 训练设置

    data_processor = DialogueDataProcessor(file_path, sp_path, block_size)
    dataset = DialogueDataset(data_processor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,   \
                                num_workers=4, pin_memory=True)
    # dataset @ dataloader
    
    train(epoch=epoch, dataloader=dataloader, model=model,   \
            device=device, writer=writer, rating=rating, wr_name=name, steps=step)

    torch.save(model.state_dict(), 'test_tower_alpha')
