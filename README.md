# tower_gpt
一个非常简单的gpt实现<br>
模型使用pytorch实现，不依赖transformers框架<br>
分词器使用sentencepiece<br>
包含了详细的中文注释，适合初学者学习使用

~~除了学习使用外实用性基本为零~~

## 配置需求
torch2.0或更高<br>
sentencepiece库<br>
galore-torch库<br>

推荐使用python3.11

## 代码
### decoder
包含了模型、训练、运行的代码
### tokenizer
包含了使用sentencepiece构建的分词器<br>

|指标|spm_dict|spm_dict_v2|spm_dict_v2.1(尚未上传)|
|    :----:   |    :----:   |     :----:    |     :----:    |
|语言|中文|中文|中文|
|训练数据大小|2GB|1.2GB|1.47GB|
|训练数据类型|新闻|书籍|新闻、书籍|
|词表大小|78336|78336|78336|
|emoji|不支持|支持|支持|

在其它条件相同时使用不同分词器的训练损失：<br>
[spm_dict](https://github.com/midway2333/tower_gpt/blob/main/png_box/v1.png)<br>
[spm_dict_v2](https://github.com/midway2333/tower_gpt/blob/main/png_box/v2.png)<br>

相较于spm_dict，v2/v2.1的训练语料更加分散<br>
但貌似spm_dict的分词效果更好

### tools
开发时使用的工具文件

## 使用
[training](https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py)文件用于训练长文本<br>
[talk_training](https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py)文件用于训练对话文本<br>
送入json文件进行训练

## 已知问题
<details close> 
<summary>  <b>已修复</b> </summary>
线性层与词向量权重共享不可用，代码以注释形式保留<br>
<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py">tfer_dataloader</a>可能导致在同一个epoch中重复利用近似文本，部分训练文本无法利用的问题 此问题可以通过使用更新的<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py">training</a>文件进行训练来避免<br>
mask无法识别padding并处理<br/>
</details>

<details open> 
<summary>  <b>尚存在</b> </summary>
两份推理文件过于老旧，无法正常运行新训练的模型<br/>
</details>

## 更新

<details close> 
<summary>  <b>8.31更新</b> </summary>
- 上传了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py">training</a>文件，新文件修复了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py">tfer_dataloader</a>中的问题，同时为训练中添加了梯度裁剪，提高了代码在大数据量下的训练效率<br/>
</details>

<details close> 
<summary>  <b>9.28更新</b> </summary>
- 上传了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py">talk_training</a>文件，新文件可以进行对话训练<br>
- 为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py">training</a>添加了梯度累计与混合精度，<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_training.py">talk_training</a>同样具有这些新特性<br>
- 删除了过时训练文件<br/>
</details>

<details close> 
<summary>  <b>10.2更新</b> </summary>
- 对<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_chat.py">模型文件</a>部分更改/优化<br>
- 修复线性层与词向量权重共享不可用的问题<br>
- 完善padding处理机制<br>
- 更换激活函数<br>
- 上传了新的工具文件<br/>
</details>

<details close> 
<summary>  <b>11.24更新</b> </summary>
- 上传了更新的<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py">training</a>文件，此文件为实验性长文本训练文件<br>
- 优点是可以直接使用json训练，可以记录训练日志<br>
- 缺点是dataset对ram需求更大，dataset效率更低<br/>
</details>

<details open> 
<summary>  <b>12.1更新</b> </summary>
- 完善了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/training.py">training</a>，现在此文件可以完成断点续训，并保存训练日志<br>
- 新的tokenizer<br>
- 删除了部分已不需要工具文件<br>
- 增加了新的工具文件<br/>
</details>
