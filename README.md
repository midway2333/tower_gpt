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

推荐使用python3.11<br>
或者使用[subsoil](https://github.com/midway2333/subsoil)快速设置开发环境
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

在其它条件相同时使用不同分词器的训练损失：(左为spm_dict，右为spm_dict_v2)<br>
![spm_dict](https://github.com/midway2333/tower_gpt/blob/main/png_box/v1.png)
![spm_dict_v2](https://github.com/midway2333/tower_gpt/blob/main/png_box/v2.png)<br>

相较于spm_dict，v2/v2.1的训练语料更加分散<br>
但貌似spm_dict的分词效果更好

### tools
开发时使用的工具文件

## 使用
[train](https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py)文件用于训练长文本<br>
[talk_train](https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py)文件用于训练对话文本<br>
送入jsonl文件进行训练<br>

~~<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/generator_train.py">generator_train</a>文件用于大数据集训练(1GB以上)~~<br>
~~送入jsonl文件进行训练~~<br>
已弃用

## 已知问题
<details close> 
<summary>  <b>已修复</b> </summary>
- 线性层与词向量权重共享不可用，代码以注释形式保留<br>
- <a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py">tfer_dataloader</a>可能导致在同一个epoch中重复利用近似文本，部分训练文本无法利用的问题<br>
- mask无法识别padding并处理<br>
-<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_train</a>缺失梯度清除<br>
- 断点续训会在开始时出现loss增加的情况<br>
- <a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dataset.py">dataset</a>在加载器下无法遍历完整数据的问题<br/>
</details>

<details open> 
<summary>  <b>尚存在</b> </summary>
- 此版本所有问题均已修复(`・ω・´)<br/>
</details>

<details open> 
<summary>  <b>其他</b> </summary>
- 推理代码随机性控制不足(只有温度采样)<br/>
</details>

## 更新

<details close> 
<summary>  <b>8.31更新</b> </summary>
- 上传了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">training</a>文件，新文件修复了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/tfer_dataloader.py">tfer_dataloader</a>中的问题，同时为训练中添加了梯度裁剪，提高了代码在大数据量下的训练效率<br/>
</details>

<details close> 
<summary>  <b>9.28更新</b> </summary>
- 上传了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_training</a>文件，新文件可以进行对话训练<br>
- 为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">training</a>添加了梯度累计与混合精度，<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_training</a>同样具有这些新特性<br>
- 删除了过时训练文件<br/>
</details>

<details close> 
<summary>  <b>10.2更新</b> </summary>
- 对<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/model.py">模型文件</a>部分更改/优化<br>
- 修复线性层与词向量权重共享不可用的问题<br>
- 完善padding处理机制<br>
- 更换激活函数<br>
- 上传了新的工具文件<br/>
</details>

<details close> 
<summary>  <b>11.24更新</b> </summary>
- 上传了更新的<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">training</a>文件，此文件为实验性长文本训练文件<br>
- 优点是可以直接使用json训练，可以记录训练日志<br>
- 缺点是dataset对ram需求更大，dataset效率更低<br/>
</details>

<details close> 
<summary>  <b>12.1更新</b> </summary>
- 完善了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">training</a>，现在此文件可以完成断点续训，并保存训练日志<br>
- 新的tokenizer<br>
- 删除了部分已不需要工具文件<br>
- 增加了新的工具文件<br/>
</details>

<details close> 
<summary>  <b>12.15更新</b> </summary>
- 对部分文件的重命名<br>
- 为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>添加了测试集支持<br>
- 修复了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_train</a>缺失梯度清除的问题<br>
- <del>我个若知居然4个月没发现这个问题</del> <br>
- 对<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_train</a>的代码优化
</details>

<details close> 
<summary>  <b>12.28更新</b> </summary>
- 拆分<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dataset.py">dataset</a>，使代码结构更清晰<br>
- 添加了使用生成器的<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/generator_train.py">generator_train</a>，适用于大数据集的训练，防止内存泄漏<br>
- 为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>添加了微调支持<br>
- 优化了<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>的log与断点续训<br>
- 代码注释优化<br/>
</details>

<details close> 
<summary>  <b>25.1.5更新</b> </summary>
- 为<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/generator_train.py">generator_train</a>添加优化器续存支持(修复断点续训会在开始时出现loss增加的情况)<br>
- 修改<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/model.py">模型文件</a>的Post layer normalization为Pre layer normalization<br>
- 修改了模型预设参数<br>
- 代码注释优化<br/>
</details>

<details close> 
<summary>  <b>25.3.2更新</b> </summary>
- 重构<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/train.py">train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_train</a>代码文件，增加了一些新功能<br>
- 修复<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dataset.py">dataset</a>在加载器下无法遍历完整数据的问题<br>
- <a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dataset.py">dataset</a>添加对</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/talk_train.py">talk_train</a>的数据加载支持<br>
- 新添<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dpo_train.py">dpo_train</a>与<a href="https://github.com/midway2333/tower_gpt/blob/main/decoder/dpo_dataset.py">dpo_dataset</a>，旨在支持dpo训练<br>
- 工具文件结构优化<br>
</details>

<details open> 
<summary>  <b>25.9.6更新</b> </summary>
- 剔除了一些不必要的线性偏置<br>
- 修复训练文件的进度条预测时间显示问题<br>
- <del>我就说项目还会更新的吧.jpg</del> <br>
</details>

## 有关问题
本项目虽然已经经过了一段时间的发展，但规范性与实用性依然无法看齐业界内的其它项目；如果您发现了任何新的问题或有关本项目的优化，欢迎在本项目的[issues](https://github.com/midway2333/tower_gpt/issues)中提出<br>
[第二代Tower](https://github.com/midway2333/Tower2)已经开源！但本项目依然维持更新

