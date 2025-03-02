import torch, json
from decoder.model import transformer
import sentencepiece as spm
from torch import Tensor   # type: ignore
from datasketch import MinHash, MinHashLSH


class DPO_clean():
    def __init__(self, model_path, sp_path, device, output_path, delta=1.5, max_sim=0.5):
        """
        清洗dpo数据

        参数:
        - model_path: 模型文件
        - sp_path: 分词器文件
        - device: 设备
        - data_path: 输入路径
        - output_path: 输出路径
        - delta: 允许保留的最小logp差值 
        - max_sim: 允许保留的最大相似度
        """

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)   # type: ignore
        self.vocab_size = self.sp.GetPieceSize()
        self.padding_id = self.sp.pad_id()
        self.eos_id = [self.sp.eos_id()]
        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        # 加载词表

        self.device = device           # 设备获取
        self.output = output_path      # 输出路径
        self.delta = delta             # 最小差值
        self.max_sim = max_sim         # 最大相似

        self.model = transformer(vocab_size=self.vocab_size, padding_idx=self.padding_id)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # 加载模型

    def tokenize(self, input, answer):
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

    def get_sim(self, st_a: str, st_b: str) -> float:
        '''获取两个句子的Dice相似度'''
        set_a, set_b = set(st_a), set(st_b)
        total_len = len(set_a) + len(set_b)
        
        if total_len == 0:
            return 0.0
        inter_set =  set_a & set_b
        return (2 * len(inter_set)) / total_len

    def logp_clean(self, data):
        """使用原模型计算logp清洗数据"""
        clean_data = []   # 用于存储已经清洗过的数据

        for idx, sample in enumerate(data):
            prompt = sample.get("prompt", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")

            chosen_line, c_mask = self.tokenize(prompt, chosen)
            rejected_line, r_mask = self.tokenize(prompt, rejected)

            c_logp = self.compute_logps(self.model, chosen_line, c_mask)
            r_logp = self.compute_logps(self.model, rejected_line, r_mask)
            # 计算logp

            diff = c_logp - r_logp
            # 计算差值

            print(f'idx: {idx} | diff: {diff.mean()}')

            if diff >= self.delta:   # 大于等于delta时保留
                clean_entry = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }   # 构建保存格式

                clean_data.append(clean_entry)
                # 添加数据

            else:   # 小于delta时舍弃
                pass

        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            # 将清洗后的数据写入JSON文件

            data_len = len(data)
            clean_len = len(clean_data)
            # 长度统计

            print(f'清洗完成；原数据共{data_len}条，清洗后{clean_len}条')
            # 打印数据

    def sim_clean(self, data):
        """使用相似度清洗数据"""
        clean_data = []   # 用于存储已经清洗过的数据

        for idx, sample in enumerate(data):
            prompt = sample.get("prompt", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")

            sim = self.get_sim(chosen, rejected)

            if sim > self.max_sim:   # 大于阈值时舍弃
                pass

            else:   # 小于阈值时保留
                clean_entry = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }   # 构建保存格式

                clean_data.append(clean_entry)
                # 添加数据

        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            # 将清洗后的数据写入JSON文件

            data_len = len(data)
            clean_len = len(clean_data)
            # 长度统计

            print(f'清洗完成；原数据共{data_len}条，清洗后{clean_len}条')
            # 打印数据

    def len_clean(self, data):
        """使用长度清洗数据"""
        clean_data = []   # 用于存储已经清洗过的数据

        for idx, sample in enumerate(data):
            prompt = sample.get("prompt", "")
            chosen = sample.get("chosen", "")
            rejected = sample.get("rejected", "")

            clen = len(chosen)
            rlen = len(rejected)

            if rlen >= clen:   # 大于时截断并保留

                rejected = rejected[:clen]
                clean_entry = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }   # 构建保存格式

                clean_data.append(clean_entry)
                # 添加数据

            else:   # 小于阈值时保留
                clean_entry = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }   # 构建保存格式

                clean_data.append(clean_entry)
                # 添加数据

        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=4)
            # 将清洗后的数据写入JSON文件

if __name__ == "__main__":

    sp_path = 'work\\tokenizer\\tower_dict_v1.0_32768.model'
    model_path = 'tower_sft.bin'
    # 分词器/模型路径

    data_path = 'clean_dpo_train035.json'
    output_path = 'clean_dpo_train_len.json'
    # 输入/输出文件路径

    cleaner = DPO_clean(
        model_path,
        sp_path,
        'cuda',
        output_path,
        0.35,
    )   # 初始化清洗代码

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        # 加载数据

    cleaner.len_clean(dataset)
