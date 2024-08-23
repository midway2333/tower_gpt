import torch
import torch.nn.functional as fc
from torch import nn, Tensor

# 参考: https://github.com/retepViolet/Transformer-

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class self_attention(nn.Module):   # 自注意力层

    def __init__(self, d, dk): 
        # d是词向量维度,dk是映射后的维度
        super().__init__()
        self.dk = dk
        self.q = nn.Linear(d, dk)
        self.k = nn.Linear(d, dk)
        self.v = nn.Linear(d, dk)
        # 三个线性变换层
        # 将输入的词向量映射到dk维度

    def attention(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor):

        output = fc.scaled_dot_product_attention   \
                (Q, K, V, attn_mask=mask, dropout_p=0.1, is_causal=False)
        
        return output
    
        # Q是查询向量,K是键向量;将K的转置与Q相乘,得到一个矩阵
        # 矩阵中每个元素表示查询向量和对应键向量之间的相似度
        # self.dk**0.5起到调节作用,使得内积不至于太大
        # 最后归一化,dim=-1使softmax沿最后一个维度进行

    def forward(self, x:tuple):
        x, mask = x
        Q = self.q(x)   # 生成query
        K = self.k(x)   # 生成key
        V = self.v(x)   # 生成value
        return self.attention(Q, K, V, mask)   # attention计算
    

class decoder(nn.Module):

    def __init__(self, head_num, d, dk, dff):
    # head_num:注意力头数 d:输入/输出维度
    # dk:每个头的维度 dff:前馈网络内部层的维度

        super().__init__()
        self.heads = nn.ModuleList()    # 存储多头注意力机制的所有头
        # ModuleList可以看作是一个存储了多个nn.Module对象的列表

        for _ in range(head_num):
            self.heads.append(self_attention(d, dk))
            # 创建了head_num个自注意力层,并将它们添加到self.heads列表中
            # 每个自注意力层都接收d和dk作为参数

        self.o = nn.Linear(head_num * dk, d)
        # 将多头注意力机制的输出合并成一个单一的输出
        # 输入维度是head_num * dk,输出维度是d

        self.norm1 = nn.LayerNorm(d)   # 层归一化

        self.ffn = nn.Sequential(   # 前馈网络
            nn.Linear(d, dff),      # 维度变换
            nn.ReLU(),              # 激活函数
            nn.Linear(dff, d),      # 维度变换
        )

        self.norm2 = nn.LayerNorm(d)   # 前馈网络输出层归一化

    def forward(self, x:tuple):
        x, mask = x   # x为需要处理的数据
        heads_res = []   # 创建了一个空列表,用于存储每个注意力头的输出
        for head in self.heads:   # 可以并行
            heads_res.append(head((x, mask)))
            # 遍历self.heads列表中的每个注意力头
            # 并将每个头的输出添加到heads_res列表中

        a = self.o(torch.concat(heads_res, dim = -1))
        # 将heads_res列表中的所有输出沿最后一个维度(dim = -1)连接起来
        # 然后将多头注意力机制的输出合并成一个单一的输出
        # a:多头注意力机制的输出

        b = self.norm1(a+x)   # 残差连接,归一化
        # 残差连接主要用于解决梯度消失和梯度爆炸问题,从而提高网络的训练效率和性能
        # 标准表示:y=f(x)+x

        y = self.norm2(self.ffn(b)+b)   # 前馈网络,残差连接,归一化

        return (y, mask)
    

class transformer(nn.Module):   # 模型实现

    def __init__(self, decoder_num=6, head_num=8, d=512, dk=256, dff=1024, vocab_size=122880):

        super().__init__()
        self.mask = Tensor()
        self.zero_mask = Tensor()
        self.pos_code = Tensor()
        # 初始化三个张量属性

        self.d = d
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, d).to(device)   # 使用独立嵌入层

        self.decoders = nn.Sequential()   # 容器模块

        for _ in range(decoder_num):
            self.decoders.append(decoder(head_num, d, dk, dff)).to(device)
            # 添加decoder_num个解码器

        self.last_linear = nn.Linear(d, self.vocab_size).to(device)
        # 线性层,将解码器的输出映射到词汇表的大小

        # self.last_linear.weight = self.embedding.weight
        # 共享weight和embedding权重
        # 有bug,先不用了

        self.softmax = nn.Softmax(dim=-1)   # 转换为概率分布

    def get_mask(self, sequence_len):

        if not self.training:
        # 检查模型是否在训练模式

            if self.zero_mask is None or sequence_len != self.zero_mask.size(0):
                # 判断输入序列的长度是否与当前掩码长度相等

                self.zero_mask = torch.zeros(sequence_len, sequence_len).to(device)
                # 生成全零掩码

            return self.zero_mask
            # 返回新的全零掩码

        if self.mask is None or sequence_len != self.mask.size(0):
        # 判断长度

            self.mask = torch.triu(torch.full((sequence_len, sequence_len),  \
                                    float('-inf')).to(device), diagonal=1)
            # 创建上三角掩码,设置掩码遮掩

        return self.mask
        # 返回掩码

    def rope_encode(self, x:int):   # RoPE位置编码

        seq_len = x   # 获得输入序列长度

        pos = torch.arange(seq_len, dtype=torch.float).to(device)
        # 创建位置索引

        inv_freq = (1.0 / (10000 ** (torch.arange(0, self.d, 2).float() / self.d))).to(device)
        # 计算逆频率
        # 在位置编码中引入多尺度信息,确保编码数值稳定性

        sinusoid_inp = torch.einsum("i,d->id", pos, inv_freq)
        # 使用爱因斯坦求和约定
        # 生成位置编码

        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        # 连接生成结果,三角位置计算

        return pos_emb

    def forward(self, x:Tensor):

        x = x.to(device)

        sequence_len = x.shape[1]
        # 获取输入张量x的第二个维度的大小

        x = self.embedding(x)   # 使用嵌入层
 
        x = x * self.d**0.5 + self.rope_encode(sequence_len)   # type: ignore
        # 将嵌入向量乘以嵌入维度的平方根(防止嵌入值过大)
        # 添加位置编码

        y, _ = self.decoders((x, self.get_mask(sequence_len)))
        # 将带有位置编码的嵌入向量和掩码传递给解码器
        # 解码器返回的y是每个位置的输出向量
        # 将解码器的输出赋值给y,并且忽略注意力权重
        # 忽略注意力权重是因为这里只想生成文本,而不需要解释模型的行为

        y = self.last_linear(y)
        # 将解码器的输出通过最后的线性层,得到每个位置的词汇表分布

        return y
