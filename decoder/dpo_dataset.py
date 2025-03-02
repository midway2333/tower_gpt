from torch.utils.data import Dataset
import json

class DPODataset(Dataset):
    def __init__(self, data):
        with open(data, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        # 读取json文件

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.data[idx]["prompt"],
            "chosen": self.data[idx]["chosen"],
            "rejected": self.data[idx]["rejected"],
        }

def collate_fn(batch):
    """批处理"""
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch]
    }
