import torch
import numpy as np
from torch.utils.data import DataLoader
from model import transformer
from dpo_dataset import DPODataset
from galore_torch import GaLoreAdamW8bit
from torch.optim.adamw import AdamW
import sentencepiece as spm
from torch.nn import functional as fc
from torch import GradScaler, autocast, Tensor, nn   # type: ignore
from torch.utils.tensorboard import SummaryWriter   # type: ignore

class dpo_train(): 
    def __init__(
            self,
            model_path,
            sp_path,
            device,
            rating,
            max_len,
            output_path,
            tb_name,
            accumulation_steps=4,
            sft_weight=0.3,
        ):
        """
        dpo对齐训练

        参数:
        - model_path: 模型文件
        - sp_path: 分词器文件
        - device: 设备
        - rating: 学习率
        - max_len: 最大训练长度
        - output_path: 输出路径
        - tb_name: tensorboard记录名
        - accumulation_steps: 梯度累积步进
        - sft_weight: sft权重
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)   # type: ignore
        self.vocab_size = self.sp.GetPieceSize()
        self.padding_id = self.sp.pad_id()
        self.eos_id = [self.sp.eos_id()]
        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        # 加载词表

        self.writer = SummaryWriter('tr_logs')
        # 初始化tensorboard

        self.device = device           # 设备获取
        self.scaler = GradScaler()     # 梯度缩放器
        self.max_len = max_len         # 最大数据长度
        self.output = output_path      # 输出路径
        self.step = 0                  # 训练步数
        self.sft_weight = sft_weight   # sft权重

        self.accumulation_steps = accumulation_steps
        self.current_accumulation = 0  # 当前累积计数
        self.accumulated_loss = 0.0    # 累积的损失值

        self.update_steps = 0          # 记录步进
        self.logp_steps = 0            # logp记录步进
        self.tb_name = tb_name         # tb记录名

        self.train_model = transformer(vocab_size=self.vocab_size, padding_idx=self.padding_id)
        self.ref_model = transformer(vocab_size=self.vocab_size, padding_idx=self.padding_id)
        self.train_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.ref_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # 加载模型

        for param in self.ref_model.parameters():
            param.requires_grad = False
        # 冻结参数

        self.opz = GaLoreAdamW8bit(self.train_model.parameters(), lr=rating)
        # 优化器

    def tokenize(self, input, answer, max_length=128):
        """使用SentencePiece进行分词"""
        input_tokens = self.sp.encode(input)   # type: ignore
        answer_tokens = self.sp.encode(answer)   # type: ignore

        full_sequence = (
            self.user_id +
            input_tokens +
            self.eos_id +
            self.bot_id +
            answer_tokens +
            self.eos_id
        )   # 构建完整序列

        prompt_mask = torch.ones(len(full_sequence))
        position_to_set_zero = len(input_tokens) + 3
        prompt_mask[:position_to_set_zero] = 0
        # 创建掩码, +3是因为 user_id/eos_id/bot_id

        if len(full_sequence) > max_length:
            full_sequence = full_sequence[:max_length - 1] + self.eos_id
            prompt_mask = prompt_mask[:max_length]  # 同步截断掩码
        else:
            pad_len = max_length - len(full_sequence)
            full_sequence += [self.padding_id] * pad_len
            prompt_mask = torch.cat([prompt_mask, torch.zeros(pad_len, dtype=torch.float)])
        # 文本长度处理

        input_ids = torch.tensor(full_sequence, dtype=torch.long).unsqueeze(0).to(self.device)
        prompt_mask = prompt_mask.unsqueeze(0).to(self.device)
        # 转换为张量并移动到指定设备

        return input_ids, prompt_mask

    def compute_logps(self, model, input_ids: Tensor, pr_mask: Tensor):
        """计算log概率"""

        logits = model(input_ids)                        # 模型输出
        log_probs = torch.log_softmax(logits, dim=-1)    # 转化为概率分布

        shift_input_ids = input_ids[:, 1:]
        shift_log_probs = log_probs[:, :-1, :]
        # 预测下一个token

        log_probs_per_token = torch.gather(shift_log_probs, 2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
        # 计算每个位置的 log 概率
        # 预测下一个token, 目标是input_ids的下一个位置

        pr_mask = pr_mask[:, 1:]
        padding_mask = (shift_input_ids != self.padding_id).float()
        # 调整掩码, 忽略padding与prompt

        log_probs_per_token = log_probs_per_token * pr_mask
        log_probs_per_token = log_probs_per_token * padding_mask
        # 掩码处理

        denominator = (pr_mask * padding_mask).sum(dim=1)
        avg_log_probs = (log_probs_per_token).sum(dim=1) / denominator
        # 计算平均 log 概率

        return avg_log_probs

    def dpo_loss(self, train_chosen, train_rejected, ref_chosen, ref_rejected, beta=0.1, lambda_reg=5):
        """
        损失函数
        - train / ref 分别代表训练和参考模型输出的概率
        - chosen / rejected 分别代表选择和拒绝的概率
        - beta: 控制对数比率(log-ratios)的缩放程度
        - lambda_reg: 正则化项的缩放系数
        """

        log_ratios = (train_chosen - ref_chosen) - (train_rejected - ref_rejected)
        # 计算对数比率

        losses = -fc.logsigmoid(beta * log_ratios)
        total_losses = losses
        # DPO损失

        reg_term = torch.clamp(torch.log(ref_chosen / train_chosen), min=0)
        total_losses += lambda_reg * reg_term.mean()
        # DPOP正则化项

        return total_losses.mean()


    def train_step(self, batch):
        """训练步"""
        self.opz.zero_grad()
        # 清除梯度

        actual_batch_size = len(batch["prompt"])
        # 获取实际批量大小

        chosen_ids = torch.zeros((actual_batch_size, self.max_len), dtype=torch.long, device=self.device)
        rejected_ids = torch.zeros((actual_batch_size, self.max_len), dtype=torch.long, device=self.device)
        Cprompt_mask = torch.zeros((actual_batch_size, self.max_len), dtype=torch.float, device=self.device)
        Rprompt_mask = torch.zeros((actual_batch_size, self.max_len), dtype=torch.float, device=self.device)
        # 预分配内存

        for idx, (prompt, chosen, rejected) in enumerate(zip(
            batch["prompt"], batch["chosen"], batch["rejected"])):

            chosen_line, c_mask = self.tokenize(prompt, chosen, self.max_len)
            rejected_line, r_mask = self.tokenize(prompt, rejected, self.max_len)

            chosen_ids[idx] = chosen_line.squeeze(0)
            rejected_ids[idx] = rejected_line.squeeze(0)
            # 输入数据

            Cprompt_mask[idx] = c_mask.squeeze(0)
            Rprompt_mask[idx] = r_mask.squeeze(0)
            # 输入掩码

        with autocast(device_type=self.device, dtype=torch.bfloat16):
            policy_chosen = self.compute_logps(self.train_model, chosen_ids, Cprompt_mask)
            policy_rejected = self.compute_logps(self.train_model, rejected_ids, Rprompt_mask)
        # 训练模型前向

        with autocast(device_type=self.device, dtype=torch.bfloat16):
            with torch.no_grad():
                ref_chosen = self.compute_logps(self.ref_model, chosen_ids, Cprompt_mask)
                ref_rejected = self.compute_logps(self.ref_model, rejected_ids, Rprompt_mask)
        # 参考模型前向

        dpo_loss = self.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss = dpo_loss
        self.accumulated_loss += loss.item()
        # 计算损失

        self.logp_steps += 1
        self.tb_logp_writer(policy_chosen, policy_rejected)
        #logp可视化

        self.scaler.scale(loss / self.accumulation_steps).backward()   # 平均梯度
        # 精度缩放, 反向传播
        self.current_accumulation += 1
        # 累计步进

        if self.current_accumulation % self.accumulation_steps == 0:
            self.scaler.unscale_(self.opz)
            self.print_grad_stats(self.train_model)
            torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1)
            # 梯度裁剪


            self.scaler.step(self.opz)    # 优化器步骤更新
            self.scaler.update()          # 参数更新
            self.opz.zero_grad()          # 清空梯度

            loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
            self.step += 1
            self.tb_writer(loss)
            # 记录损失

            self.update_steps += 1
            # 更新记录步进

            if self.update_steps % 256 == 0:
                self.up_save_model()
         
            return loss

        return None  # 未达到累积步数时返回None
    
    def save_model(self):
        """保存模型与优化器参数"""
        torch.save(self.train_model.state_dict(), self.output)   # type: ignore   # 保存模型
        opt_dict = self.opz.state_dict()                                          # 续存优化器
        torch.save(opt_dict, f'{self.output}.opz.pth')   # type: ignore           # 保存优化器

    def up_save_model(self):
        """update保存模型与优化器参数"""
        torch.save(self.train_model.state_dict(), self.output+self.update_steps)   # type: ignore   # 保存模型
        opt_dict = self.opz.state_dict()                                           # 续存优化器
        torch.save(opt_dict, f'{self.output,self.update_steps}.opz.pth')   # type: ignore           # 保存优化器

    def tb_writer(self, loss):
        """Tensorboard可视化"""
        self.writer.add_scalar(self.tb_name, loss, self.step)
        # 写入数据

    def tb_logp_writer(self, clogp, rlogp):
        """平均logp可视化"""
        self.writer.add_scalar(f'{self.tb_name}_clog', clogp.mean(), self.logp_steps)
        self.writer.add_scalar(f'{self.tb_name}_rlog', rlogp.mean(), self.logp_steps)
        # 写入数据

    def print_grad_stats(self, model):
        """打印梯度统计信息"""
        total_norm = 0.0
        max_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm()
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
        total_norm = total_norm ** 0.5
        print(f"梯度范数 - 总: {total_norm:.4f} 最大: {max_norm:.4f}")

if __name__ == "__main__":

    sp_path = 'work\\tokenizer\\tower_dict_v1.0_32768.model'
    model_path = 'tower_sftp.bin'
    # 分词器/模型路径

    steps = 12
    # 梯度累计步进

    trainer = dpo_train(
        model_path=model_path,
        sp_path=sp_path,
        device='cuda',
        rating=1e-5,
        max_len=192,
        output_path='dpo.bin',
        tb_name='dpop',
        accumulation_steps=steps,
    )   # 初始化训练代码

    dataset = DPODataset('clean_dpo_train_len.json')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for epoch in range(1):
        total_loss = 0
        for batch in dataloader:
            loss = trainer.train_step(batch)
            if loss is not None:
                total_loss += loss
        print(f"Epoch {epoch} | Loss: {total_loss*steps/len(dataloader):.4f}")
        trainer.save_model()
    # 训练模型
