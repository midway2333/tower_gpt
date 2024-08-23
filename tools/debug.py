# 有什么问题开个新文件试一下 :)
import torch
flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

print(flash)