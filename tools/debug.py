# 有什么问题开个新文件试一下 :)
from datetime import datetime
import json

def check_point_train(log_file, block_size, batch_size, epoch, rating, step, writer, name):

    """

    - log_file: 记录文件的路径
    - block_size: 窗口大小
    - batch_size: batch大小
    - epoch: 训练轮次
    - rating: 学习率
    - step: 梯度累计步进
    - writer: tensorboard文件夹
    - name: tensorboard_log名称

    """

    # 创建一个新的记录条目
    log_entry = {
        "time": datetime.now().isoformat(),
        "block_size": block_size,
        "batch_size": batch_size,
        "epoch": epoch,
        "learning_rate": rating,
        "step": step,
        "writer": writer,
        "name": name
    }


    logs = []   # 读取现有的日志文件内容
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # 如果文件不存在或为空,则创建一个空列表

    if logs:
        last_entry = logs[-1]
        previous_epoch_sum = last_entry.get("epoch", 0)
        log_entry["epoch"] = last_entry.get("epoch", 0) + epoch
    else:
        log_entry["epoch"] = epoch
        previous_epoch_sum = 0
    # 获取最新的记录以累加epoch

    logs.append(log_entry)
    # 追加新的记录

    with open(log_file, 'w') as file:
        json.dump(logs, file, indent=4)
    # 将更新后的日志列表写回文件

    return previous_epoch_sum
    # 返回当前轮次开始时的epoch轮次

x = check_point_train(
    log_file='training.log',
    block_size=128,
    batch_size=32,
    epoch=5,
    rating=0.001,
    step=100,
    writer='./tensorboard_logs',
    name='run_1'
)
print(x, type(x))
