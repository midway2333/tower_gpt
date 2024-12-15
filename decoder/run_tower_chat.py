import torch
from decoder.model import *
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('work\\tokenizer\\spm_dict.model')   # type: ignore
vocab_size = sp.GetPieceSize()
# 构建词表


model = transformer(decoder_num=3, head_num=4, d=512, dk=256, dff=1024, vocab_size=vocab_size)
# 加载模型

model.load_state_dict(torch.load('test7m.pth', weights_only=True))

model.eval()

def sample_next_token_with_temperature(probabilities, temperature=0.7):
    # 温度控制输出

    if temperature == 0:
        return probabilities.argmax(dim=-1).item()
    # 温度=0时直接返回概率最大的结果

    probabilities = torch.pow(probabilities, 1 / temperature)
    # 对每个概率值进行调整

    probabilities = probabilities / torch.sum(probabilities)
    # 归一化

    return torch.multinomial(probabilities, num_samples=1).item()
    # 按概率大小采样


def run_model(temperature=0.3):

    history = []   # 用于存储对话历史
    user_token = sp.encode('__USER__')[1]   # type: ignore
    # 获取'__USER__'的token编码

    bot_token = sp.encode('__BOT__')[1]   # type: ignore
    # 获取'__USER__'的token编码

    print('model download successfully')

    while True:

        cnt = 0
        user_input = input()

        history.append('__USER__' + user_input + '__BOT__')   # 将用户输入添加到对话历史中
        input_text = ''.join(history)           # 拼接对话历史
        tokens = sp.encode(input_text)   # 编码   # type: ignore

        while cnt <= 50:

            x = torch.tensor(tokens)   # 将tokens列表转换为张量
            x = x.unsqueeze(0)         # 在第零维增加batch_size大小

            x = model(x)
            # 获取模型输出,输出形状为[batch_size, seq_len, vocab_size]

            x = x[:, -1, :]
            # 选择最后一个时间步的输出,形状为 [batch_size, vocab_size]
            # 第一个冒号:选择所有batch
            # 第二个冒号:选择所有vocab
         
            probabilities = torch.nn.functional.softmax(x, dim=-1)
            # 最后一个维度上应用softmax函数,将logits大小转换为概率

            pred = sample_next_token_with_temperature(probabilities.squeeze(0), temperature)
            # 移除批次维度,温度参数控制采样过程的随机性

            if pred != user_token and pred != bot_token:   # 只生成bot的对话
                tokens.append(int(pred))   # 将预测的token添加到序列中

                word = sp.decode([pred])   # 预测的token解码为字词   # type: ignore
                history.append(word)   # 将解码后的单词添加到历史记忆中
                print(word, end='')    # 打印字词
                cnt += 1

                if cnt % 20 == 0:   # 打印换行符
                    print('')       # 方便阅读

            else:   # 跳出循环
                print('')
                cnt = 0
                break

run_model()
