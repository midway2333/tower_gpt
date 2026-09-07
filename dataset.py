from torch.utils.data import Dataset, IterableDataset
from numpy.random import shuffle
import random, torch, json
import sentencepiece as spm
from typing import Optional, Tuple


class TextDataProcessor:
    def __init__(
        self,
        json_file,
        sp_model_path,
        block_size,
        buffer_size=32768,
        field='text'
    ):
        """
        初始化 TextDataProcessor 类的实例

        参数:
        - json_file (str): 包含对话数据的 JSON 文件路径
        - sp_model_path (str): SentencePiece 模型文件路径
        - block_size (int): 单个输入中的最大 token 数量
        - buffer_size (int): 缓冲区大小, 默认为 32768
        - field (str): 对话数据中包含输入文本的字段名, 默认值为'text'
        """

        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)   # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int = buffer_size   # 缓冲区大小
        self.field: str = field   # 对话数据中包含输入文本的字段名

    def _encode_with_eos_sep(self, text: str, eos: str = '<|im_end|>') -> list[int]:
        """
        将文本按 eos 分割, 每个部分编码后在末尾添加 eos_id
        """
        # 以 eos 为分隔符分割文本
        parts = text.split(eos)
        token_ids = []
        for part in parts:
            part = part.strip()
            if part:   # 编码当前片段
                encoded = self.sp.encode(part, out_type=int)  # type: ignore
                token_ids.extend(encoded)

            if part:   # 只在非空片段后加 eos
                token_ids.append(self.eos_id)
        return token_ids

    def load_and_encode_data(self):
        """
        加载并编码对话数据
        使用 jsonl 文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表
        """
        inputs = []
        targets = []

        with open(self.json_file, 'r', encoding='utf-8') as f:   # 打开 jsonl 文件
            for line in f:
                line = line.strip()
                if not line:   # 跳过空行
                    continue

                dialogue = json.loads(line)
                input_text = dialogue[self.field]   # 假设每行都有 'text' 字段

                token_ids = self._encode_with_eos_sep(input_text)
                input_ids = token_ids + [self.eos_id]
                # token_ids = [self.bos_id] + token_ids   # 现在不添加 bos_id

                if len(input_ids) > self.block_size:
                    i = random.randint(0, len(input_ids) - self.block_size - 1)
                    x_data = input_ids[i:i + self.block_size]
                    y_data = input_ids[i + 1:i + 1 + self.block_size]
                    # 随机截取一个长度为 block_size 的片段

                else:   # 短序列: 填充到 block_size
                    x_data = input_ids[:-1] + [self.padding_id] * (self.block_size - len(input_ids) + 1)
                    y_data = input_ids[1:] + [self.padding_id] * (self.block_size - len(input_ids) + 1)

                x_data = torch.tensor(x_data, dtype=torch.int32)
                y_data = torch.tensor(y_data, dtype=torch.int32)

                inputs.append(x_data)
                targets.append(y_data)

        return inputs, targets

    def data_generator(self):
        """
        使用生成器加载并编码对话数据,适用于大数据集加载
        使用 jsonl 文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表
        - None: 占位符, 用于与对话数据加载模式保持一致
        """

        with open(self.json_file, 'r', encoding='utf-8') as f:
            buffer_list = []   # 缓冲区

            for line in f:   # 加载jsonl文件
                dialogue = json.loads(line.strip())

                input = dialogue[self.field]   # 获取用户输入文本
                input_ids = self._encode_with_eos_sep(input)
                # input_ids = [self.bos_id] + input_ids   # 可选添加 bos

                if len(input_ids) > self.block_size:   # 随机选择一个起始索引
                    i = random.randint(0, len(input_ids) - self.block_size - 1)
                    x_data = input_ids[i:i+self.block_size]
                    y_data = input_ids[i+1:i+1+self.block_size]

                    x_data = torch.tensor(x_data, dtype=torch.int32)
                    y_data = torch.tensor(y_data, dtype=torch.int32)
                    # 将编码信息转换为tensor    

                else:   # 短序列: 填充
                    x_data = input_ids[:-1] + [self.padding_id] * (self.block_size - len(input_ids) + 1)
                    y_data = input_ids[1:] + [self.padding_id] * (self.block_size - len(input_ids) + 1)

                    x_data = torch.tensor(x_data, dtype=torch.int32)
                    y_data = torch.tensor(y_data, dtype=torch.int32)
                    # 将编码信息转换为tensor

                buffer_list.append((x_data, y_data))   # 添加到缓冲区
                
                # 检查缓冲区是否已满
                if len(buffer_list) >= self.buffer_size:
                    random.shuffle(buffer_list)
                    for inputs, targets in buffer_list:
                        yield inputs, targets, None
                    buffer_list = []   # 清空缓冲区
            
            # 处理剩余的缓冲区数据
            if buffer_list:
                random.shuffle(buffer_list)
                for inputs, targets in buffer_list:
                    yield inputs, targets, None
    
    def data_length(self):
        """返回数据集的长度"""
        with open(self.json_file, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)


class TextDataset(Dataset):   # 负责加载和编码数据的实例
    def __init__(self, processor: TextDataProcessor):
        """初始化 TextDataset 类的实例"""
        self.inputs, self.targets = processor.load_and_encode_data()

    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)

    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx], None

class GeneratorTextDataset(IterableDataset):   # 负责生成器模式下加载和编码数据的实例
    def __init__(self, processor: TextDataProcessor):
        """初始化 GeneratorTextDataset 类的实例"""
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
    
def text_collate_fn(batch):
    """
    处理 (x, y, None) 的 batch
    返回:
    - (x_stacked, y_stacked, None)
    """
    xs, ys, _ = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    # 堆叠 x 和 y

    return xs, ys, None

""" ------------------------------------- 以上普通文本dataset ------------------------------------- """

""" ------------------------------------- 以下多轮对话dataset ------------------------------------- """

class MultiTurn_DialogueDataProcessor:
    """多轮对话数据处理器"""
    def __init__(self, json_file, sp_model_path, block_size, buffer_size=32768):
        """
        初始化 MultiTurn_DialogueDataProcessor 类的实例, 接受标准格式的多轮对话数据

        接受格式:
        - {"messages": [{"role": "user", "content": "用户输入"}, {"role": "assistant", "content": "助手回复"}]}

        具体支持:

        ```
        - 普通对话:
            {"role": "user", "content": "..."}
            {"role": "assistant", "content": "..."}

        - system:
            {"role": "system", "content": "..."}

        - reasoning:
            {
                "role": "assistant",
                "reasoning": {
                    "type": "think" / "brief_think",
                    "content": "..."
                },
                "content": "..."
            }

        - function call:
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "香港"
                            }
                        }
                    }
                ]
            }

        - tool response:
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "name": "get_weather",
                "content": {...}
            }
        ```

        字段:
        - messages (list): 包含多轮对话消息的列表,每个消息是一个字典,包含"role"和"content"键

        参数:
        - json_file (str): 包含对话数据的 jsonl 文件路径
        - sp_model_path (str): SentencePiece 模型文件路径
        - block_size (int): 单个输入中的最大 token 数量
        - buffer_size (int): 缓冲区大小, 用于批量加载数据, 默认值为 32768
        """

        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)    # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int = buffer_size   # 添加缓冲区大小

        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        self.system_id = [self.sp.PieceToId('<system>')]
        # 角色识别符

        self.think_id = [self.sp.PieceToId('<think>')]
        self.think_end_id = [self.sp.PieceToId('</think>')]
        # 思考字段识别符

        self.brief_think_id = [self.sp.PieceToId('<brief_think>')]
        self.brief_think_end_id = [self.sp.PieceToId('</brief_think>')]
        # 简短思考字段识别符

        self.tool_call_id = [self.sp.PieceToId('<tool_call>')]
        self.tool_call_end_id = [self.sp.PieceToId('</tool_call>')]
        # 工具调用字段识别符

        self.tool_response_id = [self.sp.PieceToId('<tool_response>')]
        self.tool_response_end_id = [self.sp.PieceToId('</tool_response>')]
        # 工具调用响应字段识别符

        self.newline_id = [self.sp.PieceToId('\n')]
        # 获取换行符的 id, 用于分隔对话

    def _process_single_dialogue(
        self,
        record: dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """处理单个多轮对话数据"""
        messages = record.get("messages") or record.get("dialogue") or record.get("conversations")   # 尝试不同的字段名

        if not isinstance(messages, list) or len(messages) == 0:
            return None   # 输入验证检查

        token_ids = []
        loss_mask = []   # 损失掩码, 0表示忽略loss, 1表示计算loss

        # ================ 辅助函数 ================
        def encode_text(text: dict | list | str) -> list[int]:
            """编码普通文本；dict/list 等结构自动转换为紧凑 JSON"""
            if text is None:
                return []

            if not isinstance(text, str):
                try:
                    text = json.dumps(
                        text,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )   # JSON 序列化

                except (TypeError, ValueError):   # 兜底
                    text = str(text)

            return self.sp.encode(text, out_type=int)  # type: ignore

        def append_context_masked_message(
            role_id: list[int],
            content: dict | list | str,
        ):
            """
            添加不参与 loss 的上下文消息

            标准 ChatML 格式

            参数:
            - role_id (list[int]): 角色识别符的 token id
            - content (dict | list | str): 消息内容
            """
            encoded = encode_text(content)

            ids = (
                [self.bos_id]
                + role_id
                + self.newline_id
                + encoded
                + [self.eos_id]
                + self.newline_id
            )

            token_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        def append_context_masked_ids(
            role_id: list[int],
            content_ids: list[int],
        ):
            """添加已经编码完成的不参与 loss 的上下文消息

            参数:
            - role_id (list[int]): 角色识别符的 token id
            - content_ids (list[int]): 消息内容的 token id 列表
            """

            ids = (
                [self.bos_id]
                + role_id
                + self.newline_id
                + content_ids
                + [self.eos_id]
                + self.newline_id
            )

            token_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        def serialize_tool_call(call: dict) -> Optional[str]:
            """
            将不同格式的 function call 统一为:

            {
                "name": "...",
                "arguments": {...}
            }
            """

            if not isinstance(call, dict):
                return None

            function = call.get("function")
            # OpenAI 格式:
            # {
            #   "type": "function",
            #   "function": {
            #       "name": "...",
            #       "arguments": ...
            #   }
            # }

            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", {})

            else:
                name = call.get("name")
                arguments = call.get(
                    "arguments",
                    call.get("parameters", {}),
                )

                # 简化格式:
                # {
                #   "name": "...",
                #   "arguments": {...}
                # }

            if not isinstance(name, str) or not name:
                return None

            if isinstance(arguments, str):   # 有些数据集把 arguments 存成 JSON 字符串
                try:
                    arguments = json.loads(arguments)

                except json.JSONDecodeError:   # 非法 JSON 时保留原字符串
                    pass

            normalized = {
                "name": name,
                "arguments": arguments,
            }

            return json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            )   # JSON 序列化

        # ================ tools 结构 ================
        tools = record.get("tools")
        tools_prompt: Optional[str] = None

        if tools:
            if isinstance(tools, str):
                tools_text = tools
            else:
                tools_text = json.dumps(
                    tools,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            tools_prompt = (
                "You may call one or more functions to assist with the user query." + "\n"
                "\n"
                "You are provided with function signatures within <tools></tools> XML tags:" + "\n"
                "<tools>" + "\n"
                + tools_text + "\n"
                "</tools>" + "\n"
                "\n"
                "The format of each function call must be as follows:" + "\n"
                "within "
            )

            tools_format_prompt = (
                '{"name":<function-name>,"arguments":<args-json-object>}'
            )

            tools_prompt_ids = (
                encode_text(tools_prompt)
                + self.tool_call_id
                + self.tool_call_end_id
                + encode_text(" XML tags:" + "\n")
                # within <tool_call></tool_call> XML tags:

                + self.tool_call_id
                + self.newline_id
                + encode_text(tools_format_prompt)
                + self.newline_id
                + self.tool_call_end_id
            )   # 因为 两个 tool_call 是控制符, 所以需要手动构建

        has_system = any(
            isinstance(msg, dict)
            and msg.get("role") == "system"
            for msg in messages
        )   # 判断原始 messages 是否已有 system

        tools_injected = False
     
        if tools_prompt is not None and not has_system:
            append_context_masked_ids(
                self.system_id,
                tools_prompt_ids,
            )   # 如果没有 system, 则创建一个 system 用于提供 tools

            tools_injected = True
            # 只在存在 tools 时注入到 system 中

        # ================ 消息结构 ================
        for msg in messages:   # 检查消息结构
            if not isinstance(msg, dict):
                continue   # 非字典类型, 跳过

            role = msg.get("role")   # 获取角色类别

            # ======== system ========
            if role == "system":
                content = msg.get("content")

                if content is None:
                    content = ""

                if not isinstance(content, str):
                    continue

                if tools_prompt is not None and not tools_injected:
                    # 确保 tools 存在且只注入一次

                    content_ids = encode_text(content)
                    # 编码

                    if content:
                        content_ids += self.newline_id
                        content_ids += self.newline_id
                        # 换行两次, 用于分隔 system 消息和 tools 提示

                    content_ids += tools_prompt_ids
                    append_context_masked_ids(
                        self.system_id,
                        content_ids,
                    )   # tools 只注入第一个 system

                    tools_injected = True
                    # 标记为已注入 tools

                else:
                    append_context_masked_message(
                        self.system_id,
                        content,
                    )   # 添加 system 消息到 context 中

            # ======== user ========
            elif role == "user":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue

                append_context_masked_message(
                    self.user_id,
                    content,
                )   # 添加 user 消息到 context 中

            # ======== assistant ========
            elif role == "assistant":
                assistant_ids: list[int] = []

                # ======== reasoning ========
                reasoning = msg.get("reasoning")
                reasoning_type = None
                reasoning_content = None

                if isinstance(reasoning, dict):    # 推荐的 canonical 格式
                    reasoning_type = reasoning.get("type", "think")   # 默认 think
                    reasoning_content = reasoning.get("content")

                elif isinstance(reasoning, str):
                    reasoning_type = "think"
                    reasoning_content = reasoning
                # 兼容 reasoning 直接是字符串

                elif isinstance(msg.get("think"), str):
                    reasoning_type = "think"
                    reasoning_content = msg.get("think")
                # 兼容其它数据集的旧字段

                elif isinstance(msg.get("brief_think"), str):
                    reasoning_type = "brief_think"
                    reasoning_content = msg.get("brief_think")

                elif isinstance(msg.get("reasoning_content"), str):
                    reasoning_type = "think"
                    reasoning_content = msg.get("reasoning_content")

                if isinstance(reasoning_content, str) and reasoning_content:
                    reasoning_ids = encode_text(
                        reasoning_content
                    )   # 编码 reasoning 内容

                    if reasoning_type == "brief_think":
                        assistant_ids += (
                            self.brief_think_id
                            + self.newline_id
                            + reasoning_ids
                            + self.newline_id
                            + self.brief_think_end_id
                            + self.newline_id
                        )

                    else:   # 未知 reasoning type 也默认按 think 处理
                        assistant_ids += (
                            self.think_id
                            + self.newline_id
                            + reasoning_ids
                            + self.newline_id
                            + self.think_end_id
                            + self.newline_id
                        )

                # ======== tools call ========
                tool_calls = msg.get("tool_calls")

                if tool_calls is None:   # 兼容旧式单个 function_call
                    function_call = msg.get("function_call")

                    if isinstance(function_call, dict):
                        tool_calls = [function_call]

                if isinstance(tool_calls, dict):
                    tool_calls = [tool_calls]
                    # 单 dict 也转成 list

                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        call_text = serialize_tool_call(call)

                        if call_text is None:
                            continue

                        call_ids = encode_text(call_text)

                        assistant_ids += (
                            self.tool_call_id
                            + self.newline_id
                            + call_ids
                            + self.newline_id
                            + self.tool_call_end_id
                            + self.newline_id
                        )

                # ======== 普通回答 content ========
                content = msg.get("content")
                if isinstance(content, str) and content:
                    assistant_ids += encode_text(content)

                if len(assistant_ids) == 0:
                    continue

                token_ids += (
                    [self.bos_id]
                    + self.bot_id
                    + self.newline_id
                    + assistant_ids
                    + [self.eos_id]
                    + self.newline_id
                )   # 构建 assistant 消息格式

                loss_mask += [0] * 3
                # <|im_start|> + <bot> + \n 不计算 loss

                loss_mask += [1] * (
                    len(assistant_ids) + 1
                )   # assistant body + <|im_end|> 计算 loss

                loss_mask += [0]
                # 最后的 \n 不计算 loss

            # ======== tool response ========
            elif role in ("tool", "function"):
                content = msg.get("content")
                if content is None:
                    continue

                response_ids = encode_text(content)

                token_part = (
                    [self.bos_id]
                    + self.tool_response_id
                    + self.newline_id
                    + response_ids
                    + self.newline_id
                    + self.tool_response_end_id
                    + [self.eos_id]
                    + self.newline_id
                )   # 构建 tool response 消息格式

                token_ids += token_part
                loss_mask += [0] * len(token_part)
                # tool response 是环境输入, 不计算 loss

        # ======== 截断 / 填充 ========
        if len(token_ids) > self.block_size:
            token_ids = token_ids[-(self.block_size + 1):]
            loss_mask = loss_mask[-(self.block_size + 1):]

            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
            loss_mask = loss_mask[1:]
            # 从后向前截断到 block_size

        else:
            pad_len = self.block_size - len(token_ids) + 1

            input_ids = token_ids[:-1] + [self.padding_id] * pad_len
            target_ids = token_ids[1:] + [self.padding_id] * pad_len
            loss_mask = loss_mask[1:]+ [0] * pad_len
            # 填充到 block_size

        # ======== 数据有效性检查 ========
        if len(token_ids) < 2:
            return None

        if not any(loss_mask):
            return None

        return (
            torch.tensor(input_ids, dtype=torch.int32),
            torch.tensor(target_ids, dtype=torch.int32),
            torch.tensor(loss_mask, dtype=torch.int32),
        )

    def load_and_encode_data(self):
        """小数据集, 一次性加载全部"""
        inputs, targets, loss_masks = [], [], []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue
                # 跳过空行

                record = json.loads(line)
                # 读取jsonl文件

                result = self._process_single_dialogue(record)
                # 处理单轮对话数据

                if result:
                    inputs.append(result[0])
                    targets.append(result[1])
                    loss_masks.append(result[2])

        return inputs, targets, loss_masks

    def data_generator(self):
        """用于大数据集：生成器 + 缓冲打乱"""
        buffer = []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue
                # 跳过空行

                record = json.loads(line)
                # 读取jsonl文件

                result = self._process_single_dialogue(record)
                # 处理单轮对话数据

                if not result:
                    continue

                if len(buffer) < self.buffer_size:
                    buffer.append(result)
                # 缓冲区未满, 继续添加

                else:
                    shuffle(buffer)
                    yield from buffer
                    buffer = [result]
                # 缓冲区满, 打乱并 yield, 清空

        if buffer:
            shuffle(buffer)
            yield from buffer
        # 处理剩余数据

    def data_length(self):
        """返回文件行数"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())


class Talk_DialogueDataset(Dataset):
    """用于小数据集的数据加载器"""
    def __init__(self, processor: MultiTurn_DialogueDataProcessor):
        self.inputs, self.targets, self.loss_masks = processor.load_and_encode_data()
    
    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)
    
    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx], self.loss_masks[idx]

class Talk_GeneratorDialogueDataset(IterableDataset):
    """用于大数据集的数据加载器"""
    def __init__(self, processor: MultiTurn_DialogueDataProcessor):
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
    
""" ------------------------------------- 以上多轮对话dataset ------------------------------------- """

""" ------------------------------------- 以下强化学习dataset ------------------------------------- """

class RLDataProcessor:
    """强化学习数据处理器, 处理包含 chosen 和 rejected 字段的 JSONL 文件"""
    def __init__(self, json_file, sp_model_path, block_size, buffer_size=32768):
        """
        参数:
        - json_file (str): JSONL 文件路径
        - sp_model_path (str): SentencePiece 模型路径
        - block_size (int): 序列最大长度
        - buffer_size (int): 生成器缓冲区大小

        数据格式:
        ```
        {
            "chosen": {
                "tools": [...],
                "messages": [...]
            },
            "rejected": {
                "tools": [...],
                "messages": [...]
            }
        }   # 在同一行
        ```

        支持 function call 与 reasoning
        """
        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)                 # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size = buffer_size

        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        # 获取 user 和 bot 的 id

        self.system_id = [self.sp.PieceToId('<system>')]
        # 角色识别符

        self.think_id = [self.sp.PieceToId('<think>')]
        self.think_end_id = [self.sp.PieceToId('</think>')]
        # 思考字段识别符

        self.brief_think_id = [self.sp.PieceToId('<brief_think>')]
        self.brief_think_end_id = [self.sp.PieceToId('</brief_think>')]
        # 简短思考字段识别符

        self.tool_call_id = [self.sp.PieceToId('<tool_call>')]
        self.tool_call_end_id = [self.sp.PieceToId('</tool_call>')]
        # 工具调用字段识别符

        self.tool_response_id = [self.sp.PieceToId('<tool_response>')]
        self.tool_response_end_id = [self.sp.PieceToId('</tool_response>')]
        # 工具调用响应字段识别符

        self.newline_id = [self.sp.PieceToId('\n')]
        # 获取换行符的 id, 用于分隔对话

    def _process_conversation(
        self,
        record: dict
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """处理单个多轮对话数据"""
        messages = record.get("messages") or record.get("dialogue") or record.get("conversations")   # 尝试不同的字段名
        if not isinstance(messages, list) or len(messages) == 0:
            return None   # 输入验证检查

        token_ids = []
        loss_mask = []   # 损失掩码, 0表示忽略loss, 1表示计算loss

        # ================ 辅助函数 ================
        def encode_text(text: dict | list | str) -> list[int]:
            """编码普通文本；dict/list 等结构自动转换为紧凑 JSON"""
            if text is None:
                return []

            if not isinstance(text, str):
                try:
                    text = json.dumps(
                        text,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )   # JSON 序列化

                except (TypeError, ValueError):   # 兜底
                    text = str(text)

            return self.sp.encode(text, out_type=int)  # type: ignore

        def append_context_masked_message(
            role_id: list[int],
            content: dict | list | str,
        ):
            """
            添加不参与 loss 的上下文消息

            标准 ChatML 格式

            参数:
            - role_id (list[int]): 角色识别符的 token id
            - content (dict | list | str): 消息内容
            """
            encoded = encode_text(content)

            ids = (
                [self.bos_id]
                + role_id
                + self.newline_id
                + encoded
                + [self.eos_id]
                + self.newline_id
            )

            token_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        def append_context_masked_ids(
            role_id: list[int],
            content_ids: list[int],
        ):
            """添加已经编码完成的不参与 loss 的上下文消息

            参数:
            - role_id (list[int]): 角色识别符的 token id
            - content_ids (list[int]): 消息内容的 token id 列表
            """

            ids = (
                [self.bos_id]
                + role_id
                + self.newline_id
                + content_ids
                + [self.eos_id]
                + self.newline_id
            )

            token_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

        def serialize_tool_call(call: dict) -> Optional[str]:
            """
            将不同格式的 function call 统一为:

            {
                "name": "...",
                "arguments": {...}
            }
            """

            if not isinstance(call, dict):
                return None

            function = call.get("function")
            # OpenAI 格式:
            # {
            #   "type": "function",
            #   "function": {
            #       "name": "...",
            #       "arguments": ...
            #   }
            # }

            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", {})

            else:
                name = call.get("name")
                arguments = call.get(
                    "arguments",
                    call.get("parameters", {}),
                )

                # 简化格式:
                # {
                #   "name": "...",
                #   "arguments": {...}
                # }

            if not isinstance(name, str) or not name:
                return None

            if isinstance(arguments, str):   # 有些数据集把 arguments 存成 JSON 字符串
                try:
                    arguments = json.loads(arguments)

                except json.JSONDecodeError:   # 非法 JSON 时保留原字符串
                    pass

            normalized = {
                "name": name,
                "arguments": arguments,
            }

            return json.dumps(
                normalized,
                ensure_ascii=False,
                separators=(",", ":"),
            )   # JSON 序列化

        # ================ tools 结构 ================
        tools = record.get("tools")
        tools_prompt: Optional[str] = None

        if tools:
            if isinstance(tools, str):
                tools_text = tools
            else:
                tools_text = json.dumps(
                    tools,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

            tools_prompt = (
                "You may call one or more functions to assist with the user query." + "\n"
                "\n"
                "You are provided with function signatures within <tools></tools> XML tags:" + "\n"
                "<tools>" + "\n"
                + tools_text + "\n"
                "</tools>" + "\n"
                "\n"
                "The format of each function call must be as follows:" + "\n"
                "within "
            )

            tools_format_prompt = (
                '{"name":<function-name>,"arguments":<args-json-object>}'
            )

            tools_prompt_ids = (
                encode_text(tools_prompt)
                + self.tool_call_id
                + self.tool_call_end_id
                + encode_text(" XML tags:" + "\n")
                # within <tool_call></tool_call> XML tags:

                + self.tool_call_id
                + self.newline_id
                + encode_text(tools_format_prompt)
                + self.newline_id
                + self.tool_call_end_id
            )   # 因为 两个 tool_call 是控制符, 所以需要手动构建

        has_system = any(
            isinstance(msg, dict)
            and msg.get("role") == "system"
            for msg in messages
        )   # 判断原始 messages 是否已有 system

        tools_injected = False
     
        if tools_prompt is not None and not has_system:
            append_context_masked_ids(
                self.system_id,
                tools_prompt_ids,
            )   # 如果没有 system, 则创建一个 system 用于提供 tools

            tools_injected = True
            # 只在存在 tools 时注入到 system 中

        # ================ 消息结构 ================
        for msg in messages:   # 检查消息结构
            if not isinstance(msg, dict):
                continue   # 非字典类型, 跳过

            role = msg.get("role")   # 获取角色类别

            # ======== system ========
            if role == "system":
                content = msg.get("content")

                if content is None:
                    content = ""

                if not isinstance(content, str):
                    continue

                if tools_prompt is not None and not tools_injected:
                    # 确保 tools 存在且只注入一次

                    content_ids = encode_text(content)
                    # 编码

                    if content:
                        content_ids += self.newline_id
                        content_ids += self.newline_id
                        # 换行两次, 用于分隔 system 消息和 tools 提示

                    content_ids += tools_prompt_ids
                    append_context_masked_ids(
                        self.system_id,
                        content_ids,
                    )   # tools 只注入第一个 system

                    tools_injected = True
                    # 标记为已注入 tools

                else:
                    append_context_masked_message(
                        self.system_id,
                        content,
                    )   # 添加 system 消息到 context 中

            # ======== user ========
            elif role == "user":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue

                append_context_masked_message(
                    self.user_id,
                    content,
                )   # 添加 user 消息到 context 中

            # ======== assistant ========
            elif role == "assistant":
                assistant_ids: list[int] = []

                # ======== reasoning ========
                reasoning = msg.get("reasoning")
                reasoning_type = None
                reasoning_content = None

                if isinstance(reasoning, dict):    # 推荐的 canonical 格式
                    reasoning_type = reasoning.get("type", "think")   # 默认 think
                    reasoning_content = reasoning.get("content")

                elif isinstance(reasoning, str):
                    reasoning_type = "think"
                    reasoning_content = reasoning
                # 兼容 reasoning 直接是字符串

                elif isinstance(msg.get("think"), str):
                    reasoning_type = "think"
                    reasoning_content = msg.get("think")
                # 兼容其它数据集的旧字段

                elif isinstance(msg.get("brief_think"), str):
                    reasoning_type = "brief_think"
                    reasoning_content = msg.get("brief_think")

                elif isinstance(msg.get("reasoning_content"), str):
                    reasoning_type = "think"
                    reasoning_content = msg.get("reasoning_content")

                if isinstance(reasoning_content, str) and reasoning_content:
                    reasoning_ids = encode_text(
                        reasoning_content
                    )   # 编码 reasoning 内容

                    if reasoning_type == "brief_think":
                        assistant_ids += (
                            self.brief_think_id
                            + self.newline_id
                            + reasoning_ids
                            + self.newline_id
                            + self.brief_think_end_id
                            + self.newline_id
                        )

                    else:   # 未知 reasoning type 也默认按 think 处理
                        assistant_ids += (
                            self.think_id
                            + self.newline_id
                            + reasoning_ids
                            + self.newline_id
                            + self.think_end_id
                            + self.newline_id
                        )

                # ======== tools call ========
                tool_calls = msg.get("tool_calls")

                if tool_calls is None:   # 兼容旧式单个 function_call
                    function_call = msg.get("function_call")

                    if isinstance(function_call, dict):
                        tool_calls = [function_call]

                if isinstance(tool_calls, dict):
                    tool_calls = [tool_calls]
                    # 单 dict 也转成 list

                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        call_text = serialize_tool_call(call)

                        if call_text is None:
                            continue

                        call_ids = encode_text(call_text)

                        assistant_ids += (
                            self.tool_call_id
                            + self.newline_id
                            + call_ids
                            + self.newline_id
                            + self.tool_call_end_id
                            + self.newline_id
                        )

                # ======== 普通回答 content ========
                content = msg.get("content")
                if isinstance(content, str) and content:
                    assistant_ids += encode_text(content)

                if len(assistant_ids) == 0:
                    continue

                token_ids += (
                    [self.bos_id]
                    + self.bot_id
                    + self.newline_id
                    + assistant_ids
                    + [self.eos_id]
                    + self.newline_id
                )   # 构建 assistant 消息格式

                loss_mask += [0] * 3
                # <|im_start|> + <bot> + \n 不计算 loss

                loss_mask += [1] * (
                    len(assistant_ids) + 1
                )   # assistant body + <|im_end|> 计算 loss

                loss_mask += [0]
                # 最后的 \n 不计算 loss

            # ======== tool response ========
            elif role in ("tool", "function"):
                content = msg.get("content")
                if content is None:
                    continue

                response_ids = encode_text(content)

                token_part = (
                    [self.bos_id]
                    + self.tool_response_id
                    + self.newline_id
                    + response_ids
                    + self.newline_id
                    + self.tool_response_end_id
                    + [self.eos_id]
                    + self.newline_id
                )   # 构建 tool response 消息格式

                token_ids += token_part
                loss_mask += [0] * len(token_part)
                # tool response 是环境输入, 不计算 loss

        # ======== 截断 / 填充 ========
        if len(token_ids) > self.block_size:
            input_ids = token_ids[:self.block_size]
            target_ids = token_ids[1:self.block_size + 1]
            loss_mask = loss_mask[1:self.block_size + 1]
            # RL 优先保头掐尾

        else:
            pad_len = self.block_size - len(token_ids) + 1

            input_ids = token_ids[:-1] + [self.padding_id] * pad_len
            target_ids = token_ids[1:] + [self.padding_id] * pad_len
            loss_mask = loss_mask[1:]+ [0] * pad_len
            # 填充到 block_size

        # ======== 数据有效性检查 ========
        if len(token_ids) < 2:
            return None

        if not any(loss_mask):
            return None

        return (
            torch.tensor(input_ids, dtype=torch.int32),
            torch.tensor(target_ids, dtype=torch.int32),
            torch.tensor(loss_mask, dtype=torch.int32),
        )

    def load_and_encode_data(self):
        """
        一次性加载全部数据 (适用于小数据集)

        返回:
        -list: 每个元素为 (chosen_in, chosen_tar, chosen_mask, rejected_in, rejected_tar, rejected_mask)
        """
        samples = []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:   # 跳过空行
                    continue

                record: dict = json.loads(line)

                chosen = record.get("chosen")
                if not chosen:
                    continue
                chosen_result = self._process_conversation(chosen)
                if not chosen_result:
                    continue
                # 处理 chosen 对话

                rejected = record.get("rejected")
                if not rejected:
                    continue
                rejected_result = self._process_conversation(rejected)
                if not rejected_result:
                    continue
                # 处理 rejected 对话

                samples.append(chosen_result + rejected_result)   # 合并为一个样本

        return samples

    def data_generator(self):
        """
        生成器方式加载数据 (适用于大数据集)

        返回:
        - (chosen_in, chosen_tar, chosen_mask, rejected_in, rejected_tar, rejected_mask)
        """
        buffer = []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:   # 跳过空行
                    continue

                record: dict = json.loads(line)

                chosen = record.get("chosen")
                rejected = record.get("rejected")
                if not chosen or not rejected:
                    continue

                chosen_result = self._process_conversation(chosen)
                rejected_result = self._process_conversation(rejected)
                if not chosen_result or not rejected_result:
                    continue

                sample = chosen_result + rejected_result   # 六元组

                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                else:
                    random.shuffle(buffer)
                    yield from buffer
                    buffer = [sample]

        if buffer:   # 处理剩余数据
            random.shuffle(buffer)
            yield from buffer

    def data_length(self):
        """返回文件行数 (用于进度条等)"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())


class RLDataset(Dataset):
    """用于小数据集的 RL 数据集 (一次性加载)"""
    def __init__(self, processor: RLDataProcessor):
        self.samples = processor.load_and_encode_data()   # 每个样本为六元组

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # 直接返回六元组


class RLGeneratorDataset(IterableDataset):
    """用于大数据集的 RL 数据集 (生成器方式)"""
    def __init__(self, processor: RLDataProcessor):
        super().__init__()
        self.processor = processor

    def __iter__(self):
        return iter(self.processor.data_generator())


def rl_collate_fn(batch):
    """
    处理 RL 数据集的 batch, batch 中每个元素为六元组

    返回:
    - chosen_inputs, chosen_targets, chosen_masks,
    - rejected_inputs, rejected_targets, rejected_masks
    """
    chosen_inputs, chosen_targets, chosen_masks, \
    rejected_inputs, rejected_targets, rejected_masks = zip(*batch)

    chosen_inputs = torch.stack(chosen_inputs)
    chosen_targets = torch.stack(chosen_targets)
    chosen_masks = torch.stack(chosen_masks)

    rejected_inputs = torch.stack(rejected_inputs)
    rejected_targets = torch.stack(rejected_targets)
    rejected_masks = torch.stack(rejected_masks)

    return (
        chosen_inputs, chosen_targets, chosen_masks,
        rejected_inputs, rejected_targets, rejected_masks
    )
