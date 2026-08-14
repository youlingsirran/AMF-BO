# import torch
#
# def save_dataset(dataloader, filepath):
#     all_batches = []
#
#     for batch in dataloader:
#         # 根据batch的类型进行不同的处理
#         if isinstance(batch, (list, tuple)):
#             # 单个tensor
#             saved_batch = tuple(item.clone() for item in batch)
#         elif isinstance(batch, dict):
#             # 字典，处理每个键值对
#             saved_batch = {}
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     saved_batch[key] = value.clone().detach()
#                 else:
#                     saved_batch[key] = value
#
#         else:
#             # 其他数据类型（如自定义对象），尝试直接保存
#             saved_batch = batch
#
#         all_batches.append(saved_batch)
#
#     torch.save(all_batches, f"./dataloader/{filepath}.pt")
#     print(f"成功保存 {len(all_batches)} 个批次到 {filepath}")
#
# def load_dataset(filepath):
#     all_batches = torch.load(f"./dataloader/{filepath}.pt")
#     return all_batches
import torch

# from test import *
#
# sequences = [
#     torch.randn(3, 128),   # 第一个序列: 3个128维向量
#     torch.randn(1, 128),   # 第二个序列: 1个128维向量
#     torch.randn(5, 128),    # 第三个序列: 5个128维向量
#     torch.FloatTensor(),                       # 空序列
#     torch.randn(2, 128)    # 2个128维向量
# ]
#
# label = ['cat', 'dog', 'bird', 'dog', 'bird']
#
# model = get_LSTM_model(sequences, label)
#
# print(torch.FloatTensor())
# feat = model.extract(torch.FloatTensor())  # -> (32,) 全 0
# # feat = model.extract(torch.randn(3, 128))  # -> (32,) 真实特征
# print(feat)
#
# print((torch.randn(3, 128)).type())



import torch
import numpy as np

your_list = [[1, 2, 3], np.array([4, 5, 6]), [], [7, 8, 9]]
print(your_list)

tensor_list = [
    torch.FloatTensor(element) if len(element) > 0 else torch.FloatTensor()
    for element in your_list
]

print(tensor_list)