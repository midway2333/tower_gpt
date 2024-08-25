# tower_gpt
一个非常简单的gpt实现<br>
包含了详细的中文注释，适合初学者学习使用

~~除了学习使用外实用性基本为零~~

## 配置需求
torch2.0或更高<br>
sentencepiece库<br>
galore-torch库<br>

## 代码
### decoder
包含了模型、训练、运行以及dataloader代码
### tokenizer
包含了使用sentencepiece构建的分词器<br>
预设词库基于2GB中文文本训练
### tools
开发时使用的工具文件

## 使用
使用[tools](https://github.com/midway2333/tower_gpt/tree/main/tools)中的[txt_to_np](https://github.com/midway2333/tower_gpt/blob/main/tools/txt_to_np.py)文件将要使用的文本转换为numpy格式<br>
将转换后的文件送入模型训练

## 已知问题
线性层与词向量权重共享不可用，代码以注释形式保留<br>
[tfer_dataloader](https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py)可能导致在同一个epoch中重复利用近似文本，部分训练文本无法利用的问题
