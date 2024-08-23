import sentencepiece as spm
import os

# 定义语料文件路径
corpus = "dict_all.txt"

# 定义分词器的模型名称
model_prefix = "spm_dict"

# 训练分词器
spm.SentencePieceTrainer.train(     # type: ignore
    input=corpus,                   # 输入语料文件
    model_prefix=model_prefix,      # 输出模型前缀
    vocab_size=78336,               # 词汇表大小
    model_type='bpe',               # 分词模型类型
    max_sentencepiece_length=6,     # 最长单分词字数
    input_format="text",            # 输入纯文本文件
    max_sentence_length=32768,      # 训练时最大允许的句子长度
    byte_fallback=True,             # 启用字节回退机制
    character_coverage=0.9995,      # 字符覆盖率
    split_digits=True,              # 将数字拆分为单个token
    num_threads=os.cpu_count(),     # 使用的线程数
    self_test_sample_size=0,        # 自测样本的大小
    unk_surface=r" \342\201\207 ",  # 指定未知字符形式
    pad_id=3,                       # 设置pad_id    
    user_defined_symbols=['__USER__', '__BOT__']
    # 添加格外分词
)

print(f"模型已训练完成,并保存为 {model_prefix}.model和{model_prefix}.vocab")

