import json
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
import sentencepiece as spm
import torch
from galore_torch import GaLoreAdamW8bit
from torch.utils.tensorboard import SummaryWriter   # type: ignore
from model import *
from torch import GradScaler, autocast   # type: ignore
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Optional
from dataset import *

"""

训练模型的文件
我写完了才发现一个问题
我没有指定计划训练轮次
这也是启用学习调度器可能导致断点续训不正常的原因
先就这样吧

"""

class train():
    def __init__(
    self, 
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    model_path: str,
    device: torch.device,
    writer,
    rating: float,
    tb_name: str,
    wr_name,
    steps: int,
    log_file: str,
    block_size: int,
    batch_size: int,
    update_steps: int,
    data_length: int,
    train_steps: int = 0,
    history_epoch: int = 0,
    test_set: bool = False,
    test_dataloader: Optional[DataLoader] = None,
    opz_path = None,
    output_path=None,
    use_scheduler: bool = False,
    padding_id: int = 3,
    ):

        """

        训练模型的函数

        参数:
        - epoch (int): 训练的轮数
        - dataloader (DataLoader): 数据加载器
        - model (nn.Module): 要训练的模型架构
        - model_path: 模型名称
        - device (torch.device): 设备类型(CPU或GPU)
        - writer: 用于记录训练日志的对象
        - rating (float): 学习率
        - tb_name: tensorboard文件夹
        - wr_name: 训练日志名称
        - steps: 梯度累积步进
        - log_file: log文件
        - block_size: 窗口大小(用于log)
        - batch_size: 批次大小(用于log与loss计算)
        - update_steps: 更新步数
        - train_steps: 断点续训轮数
        - history_epoch: 已经训练的轮数
        - test_set: 是否运用测试集
        - test_dataloader: 测试数据加载器
        - opz_path: 优化器参数文件路径
        - output_path: 输出模型路径
        - use_scheduler: 是否启用学习率调度器
        - padding_id: 填充 id

        注意: 启用学习调度器可能导致断点续训不正常

        """

        self.epoch = epoch
        self.dataloader = dataloader
        self.model = model
        self.model_path = model_path
        self.device = device
        self.writer = writer
        self.rating = rating
        self.tb_name = tb_name
        self.wr_name = wr_name
        self.steps = steps
        self.log_file = log_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.train_steps = train_steps
        self.history_epoch = history_epoch
        self.test_set = test_set
        self.test_dataloader = test_dataloader
        self.opz_path = opz_path
        self.output_path = output_path
        self.data_length = data_length
        self.use_scheduler = use_scheduler
        self.padding_id = padding_id

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.padding_id)   # 交叉熵损失函数
        self.accumulation_steps = steps   # 设置累积步数
        self.scaler = GradScaler()   # 梯度缩放器

        self.all_epoch = self.epoch + self.history_epoch
        # 计算总轮数

        self.optimizer = GaLoreAdamW8bit(model.parameters(), lr=rating)   # 优化器

        self.output_path = output_path or model_path
        # 输出模型路径判定

        self.opz_path = opz_path or f"{model_path}_optimizer.pth"
        # 优化器参数文件路径判定
        
        if train_steps != 0:   # 断点续训加载优化器
            try:
                self.optimizer_state = torch.load(self.opz_path, weights_only=True)
                self.optimizer.load_state_dict(self.optimizer_state)
                # 加载优化器参数

            except:
                print('未找到优化器参数文件,可能影响训练')

        if use_scheduler:
            self.lr_scheduler()

    def train_model(self):
        self.progress()   # 初始化进度条

        for i in range(self.epoch):
            self.model.train()   # 设置为训练模式
            self.history_epoch += 1   # 累加历史训练轮数
            local_loss = 0   # 初始化记录loss

            epoch_show_txt = 'epoch: {}/{}'.format(
                self.history_epoch, self.all_epoch
            )   # 设置epoch更新信息

            for step, (x, y) in enumerate(self.dataloader):   # 生成步进索引

                x, y = x.to(self.device).long(), y.to(self.device).long()   # 将数据移动到指定设备上

                with autocast(device_type=str(self.device)):
                # 自动混合精度

                    pred = self.model(x)
                    pred = pred.view(-1, pred.size(-1))
                    # 将 pred 重新形状为 [batch_size * sequence_length, vocab_size]

                    y = y.view(-1)
                    # 将 y 重新形状为 [batch_size * sequence_length]

                    loss = self.loss_fn(pred, y) / self.accumulation_steps   # 计算平均损失

                self.scaler.scale(loss).backward()
                # 使用混合精度来缩放损失,反向传播

                local_loss += loss.item()
                # 计算一个批次的损失

                self.train_progress.update(self.tsp_progress, advance=1 / self.update_steps)
                # 更新 tsp 进度条

                if (step + 1) % self.accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # 梯度裁剪

                    self.scaler.step(self.optimizer)   # 更新模型参数
                    self.scaler.update()          # 调整缩放比例
                    self.optimizer.zero_grad()    # 清除之前的梯度

                    if self.use_scheduler:   # 更新学习率
                        self.rate_scheduler.step()

                if (step + 1) % self.update_steps == 0:   # 每隔一定步数更新一次日志
                    self.train_steps += 1   # 更新断点步数

                    tsp_show_txt = 'train_steps: {}/{}'.format(
                        self.train_steps, self.all_tsp
                    )   # 设置tsp更新信息

                    self.train_progress.update(self.tsp_progress, show_info=tsp_show_txt)
                    # 更新 tsp 信息

                    local_loss = (local_loss / self.update_steps) * self.accumulation_steps
                    # 计算平均损失,不必乘以batch_size,除以update_steps已经获得了平均值

                    self.writer.add_scalar(self.wr_name, local_loss, self.train_steps)
                    # 记录训练损失

                    local_loss = 0

                    train_log = log(
                    model_path=self.output_path,
                    log_file=self.log_file,
                    block_size=self.block_size,
                    batch_size=self.batch_size,
                    epoch=0,
                    train_step=self.train_steps,
                    rating=self.rating,
                    step=self.steps,
                    writer=self.tb_name,
                    tb_name=self.wr_name
                    )   # 创建日志对象

                    train_log.train_log()   # 记录训练日志
                    self.test_model()   # 测试模型
                    self.save_model()   # 保存模型

            self.train_progress.update(self.epoch_progress, show_info=epoch_show_txt, advance=1)
            # 更新epoch信息与进度条

            self.test_model()   # 测试模型
            self.epoch_save()   # 每个epoch保存模型

            train_log = log(
            model_path=self.output_path,
            log_file=self.log_file,
            block_size=self.block_size,
            batch_size=self.batch_size,
            epoch=1,
            train_step=self.train_steps,
            rating=self.rating,
            step=self.steps,
            writer=self.tb_name,
            tb_name=self.wr_name
            )   # 创建日志对象

            train_log.train_log()   # 记录训练日志

    def test_model(self):
        """运用评估集测试模型"""
        if self.test_set and self.test_dataloader is not None:
            self.model.eval()   # 设置为评估模式
            epoch_test_loss = 0

            with autocast(device_type=str(self.device)):
            # 自动混合精度

                with torch.no_grad():  # 不需要计算梯度
                    for tx, ty in self.test_dataloader:
                        tx, ty = tx.to(self.device).long(), ty.to(self.device).long()

                        test_pred = self.model(tx)
                        test_pred = test_pred.view(-1, test_pred.size(-1))
                        ty = ty.view(-1)
                        # 同上

                        test_loss = self.loss_fn(test_pred, ty)
                        epoch_test_loss += test_loss.item()
                
            self.avg_test_loss = (epoch_test_loss / len(self.test_dataloader))   # type: ignore
            # 计算损失

            self.writer.add_scalar(self.wr_name+'_test', self.avg_test_loss, self.train_steps)
            # 记录测试损失

            self.model.train()

        else:   # 无测试集时跳过
            pass

    def save_model(self):
        """保存模型与优化器参数"""
        torch.save(self.model.state_dict(), self.output_path)   # type: ignore     # 保存模型
        opt_dict = self.optimizer.state_dict()                                     # 续存优化器
        torch.save(opt_dict, self.opz_path)   # type: ignore                       # 保存优化器

        if self.use_scheduler:   # 保存调度器
            scheduler_state = self.rate_scheduler.state_dict()
            torch.save(scheduler_state, f"{self.opz_path}_scheduler.pth")

    def epoch_save(self):
        """每epoch保存模型,用于防止训练出错"""
        torch.save(self.model.state_dict(), self.output_path+'_epoch'+str(self.history_epoch))   # type: ignore
        opt_dict = self.optimizer.state_dict() 
        torch.save(opt_dict, self.opz_path+'_epoch'+str(self.history_epoch))   # type: ignore

        if self.use_scheduler:
            scheduler_state = self.rate_scheduler.state_dict()
            torch.save(scheduler_state, f"{self.opz_path}_scheduler.pth"+'_epoch'+str(self.history_epoch))

    def progress(self):
        """进度条可视化训练进度"""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),   # 显示任务的描述信息
            BarColumn(),   # 显示进度条
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),   # 设置样式,保留三位数的整数百分比,右对齐
            TimeRemainingColumn(),   # 显示基于当前进度推测估计的剩余时间
            TimeElapsedColumn(),   # 显示运行时间
            TextColumn("[bold blue]{task.fields[show_info]}"),   # 额外信息
            refresh_per_second=1,  # 每1秒钟更新一次
        )

        self.epoch_progress = progress.add_task(description='epoch: ', show_info='', total=self.epoch)
        # epoch进度条

        self.all_tsp = self.data_length * self.all_epoch //   \
            (self.batch_size * self.update_steps)
        self.tsp_progress = progress.add_task(description='steps: ', show_info='', total=self.all_tsp)
        # tsp进度条

        self.train_progress = progress   # 对象化进度条
        self.train_progress.start()   # 启动进度条

    def lr_scheduler(self):
        """学习率调度器"""
        if not self.use_scheduler:
            return

        last_step = -1 if self.train_steps == 0 else self.train_steps - 1

        self.rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=50 * self.rating,
            epochs=self.all_epoch,
            cycle_momentum=False,
            steps_per_epoch=int(np.ceil(self.data_length / (self.batch_size * self.accumulation_steps))),
            div_factor=50,
            last_epoch=last_step,
        )   # 创建学习率调度器

        # 如果是断点续训,尝试加载调度器状态
        if self.train_steps != 0:
            try:
                scheduler_state = torch.load(f"{self.opz_path}_scheduler.pth")
                self.rate_scheduler.load_state_dict(scheduler_state)
                print('成功加载学习率调度器状态')
            except:
                print('未找到学习率调度器状态文件,可能影响训练')

class log():
    def __init__(self, model_path, log_file, block_size, batch_size, epoch,   \
                    train_step, rating, step, writer, tb_name):

        """

        日志记录

        参数:
        - model_path: 模型名
        - log_file: 记录文件的路径
        - block_size: 窗口大小
        - batch_size: batch大小
        - epoch: 训练轮次
        - train_step: 训练步数
        - rating: 学习率
        - step: 梯度累计步进
        - writer: tensorboard文件夹
        - tb_name: tensorboard_log名称

        """

        self.model_path = model_path
        self.log_file = log_file
        self.block_size = block_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_step = train_step
        self.rating = rating
        self.step = step
        self.writer = writer
        self.tb_name = tb_name

    def train_log(self):
        """记录训练日志"""

        log_entry = {
            "time": datetime.now().isoformat(),
            "model": self.model_path,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "epoch": self.epoch,
            "train_step": self.train_step,
            "learning_rate": self.rating,
            "step": self.step,
            "writer": self.writer,
            "tb_name": self.tb_name
        }   # 创建一个新的记录条目

        logs = []   # 读取现有的日志文件内容
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # 如果文件不存在或为空,则创建一个空列表

        if logs:   # 获取最新的记录以累加epoch
            last_entry = logs[-1]
            log_entry["epoch"] = last_entry.get("epoch", 0) + self.epoch
        else:   # 首次创建日志文件,设置epoch为0
            log_entry["epoch"] = 0

        logs.append(log_entry)
        # 追加新的记录

        with open(self.log_file, 'w') as file:
            json.dump(logs, file, indent=4)
        # 将更新后的日志列表写回文件

    @staticmethod   # 不接受self参数
    def get_previous_train_step(log_file):
        """获取日志文件中最后一个记录的train_step轮次"""

        try:
            with open(log_file, 'r') as file:
                logs = json.load(file)
                if logs:
                    last_entry = logs[-1]
                    return last_entry.get("train_step", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass   # 如果文件不存在或为空,则返回0
        return 0
    
    @staticmethod   # 不接受self参数
    def get_previous_epoch(log_file):
        """获取日志文件中最后一个记录的epoch轮次"""

        try:
            with open(log_file, 'r') as file:
                logs = json.load(file)
                if logs:
                    last_entry = logs[-1]
                    return last_entry.get("epoch", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass   # 如果文件不存在或为空,则返回0
        return 0

def is_finetune(model: nn.Module, model_name: str):

    """

    微调判定

    参数:
    - model (nn.Module): 要微调的模型
    - model_name (str): 预训练模型的文件名

    """

    model.load_state_dict(torch.load(model_name, weights_only=True))
    # 加载预训练模型的权重

    layers_freeze = [model.embedding]
    # 指定embedding层参数

    for layer in layers_freeze:
        for param in layer.parameters():
            param.requires_grad = False
    # 冻结参数

if __name__ == '__main__':

    file_path = 'data\\test.jsonl'
    block_size = 128
    batch_size = 12
    # 训练集设置 jsonl格式

    test_file_path = 'data\\train_test.json'
    test_block_size = 128
    test_batch_size = 12
    # 测试集设置 json格式

    sp = spm.SentencePieceProcessor()
    sp_path = 'work\\tokenizer\\tower_dict_v1.0_32768.model'
    sp.load(sp_path)   # type: ignore
    vocab_size = sp.GetPieceSize()
    padding_id = sp.pad_id()
    # 构建词表

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设备获取

    model = transformer(vocab_size=vocab_size, padding_idx=padding_id)
    # 模型设置

    epoch = 10
    rating = 0.0001
    step = 16                   # 梯度累积步数
    update_steps = 512          # 更新步数
    writer_file = 'tr_logs'
    writer = SummaryWriter(writer_file)
    wr_name = 'test_pre2'     # tensorboard记录名称
    use_test = True
    fin_tuning = False   # 是否微调 # 记得调低学习率
    use_scheduler = True   # 是否启用动态学习率
    # 训练设置

    model_path = 'test2.bin'
    output_path = None   # None即覆盖原模型
    log_file = 'test2.log'
    opz_path = 'test2.bin_optimizer.pth'
    tsp = log.get_previous_train_step(log_file)
    ep  = log.get_previous_epoch(log_file)
    # 信息设置

    if tsp != 0:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    # 断点续训

    if fin_tuning:
        is_finetune(model, model_path)
        print('微调设置成功')
    # 微调

    data_processor = DialogueDataProcessor(file_path, sp_path, block_size)
    dataset = GeneratorDialogueDataset(data_processor)
    data_length = data_processor.data_length()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,   \
                                num_workers=0, pin_memory=True)
    # 训练集 dataset @ dataloader
    # 生成器加载 num_workers 只能为0

    test_data_processor = DialogueDataProcessor(test_file_path, sp_path, test_block_size)

    if use_test:
        test_dataset = DialogueDataset(test_data_processor)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,   \
                                    num_workers=8, pin_memory=True)

    else:
        test_dataloader = None
    # 训练集 dataset @ dataloader

    run = train(epoch=epoch,
        dataloader=dataloader,
        model=model,
        model_path=model_path,
        device=device,
        writer=writer,
        rating=rating,
        tb_name=writer_file,
        wr_name=wr_name,
        steps=step,
        log_file=log_file,
        block_size=block_size,
        batch_size=batch_size,
        update_steps=update_steps,
        data_length=data_length,
        train_steps=tsp,
        history_epoch=ep,
        test_set=use_test,
        test_dataloader=test_dataloader,
        opz_path=opz_path,
        output_path=output_path,
        use_scheduler=use_scheduler,
        padding_id=padding_id,
    )

    run.train_model()   # 训练模型
    writer.close()   # 关闭writer
