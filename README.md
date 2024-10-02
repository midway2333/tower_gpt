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
对于[training](https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py)文件：使用[tools](https://github.com/midway2333/tower_gpt/tree/main/tools)中的[txt_to_np](https://github.com/midway2333/tower_gpt/blob/main/tools/txt_to_np.py)文件将要使用的文本转换为numpy格式，将转换后的文件送入模型训练<br>
对于[talk_training](https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py)文件：直接送入json文件进行训练，无需转换格式

## 已知问题
~~线性层与词向量权重共享不可用，代码以注释形式保留~~ 问题已修复<br>
~~[tfer_dataloader](https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py)可能导致在同一个epoch中重复利用近似文本，部分训练文本无法利用的问题~~ 此问题可以通过使用更新的[training](https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py)文件进行训练来避免<br>
~~mask无法识别padding并处理~~ 此若知问题已修复

## 更新
### 8.31更新
上传了[training](https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py)文件，新文件修复了[tfer_dataloader](https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py)中的问题，同时为训练中添加了梯度裁剪，提高了代码在大数据量下的训练效率
### 9.28更新
上传了[talk_training](https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py)文件，新文件可以进行对话训练<br>
为[training](https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py)添加了梯度累计与混合精度，[talk_training](https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py)同样具有这些新特性<br>
删除了过时训练文件
### 10.2更新
对[模型文件](https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_chat.py)部分更改/优化<br>
修复线性层与词向量权重共享不可用的问题<br>
完善padding处理机制<br>
上传了新的工具文件
