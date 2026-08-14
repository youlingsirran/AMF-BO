# import torch
#
# def save_dataset(dataloader, filepath):
#     all_batches = []
#
#     for batch in dataloader:
#         if isinstance(batch, (list, tuple)):
#             saved_batch = tuple(item.clone() for item in batch)
#         elif isinstance(batch, dict):
#             saved_batch = {}
#             for key, value in batch.items():
#                 if isinstance(value, torch.Tensor):
#                     saved_batch[key] = value.clone().detach()
#                 else:
#                     saved_batch[key] = value
#
#         else:
#             saved_batch = batch
#
#         all_batches.append(saved_batch)
#
#     torch.save(all_batches, f"./dataloader/{filepath}.pt")
#
# def load_dataset(filepath):
#     all_batches = torch.load(f"./dataloader/{filepath}.pt")
#     return all_batches
import numpy as np
import torch

# from test import *
#
# sequences = [
# ]
#
# label = ['cat', 'dog', 'bird', 'dog', 'bird']
#
# model = get_LSTM_model(sequences, label)
#
# print(torch.FloatTensor())
# print(feat)
#
# print((torch.randn(3, 128)).type())




your_list = [[1, 2, 3], np.array([4, 5, 6]), [], [7, 8, 9]]
print(your_list)

tensor_list = [
    torch.FloatTensor(element) if len(element) > 0 else torch.FloatTensor()
    for element in your_list
]

print(tensor_list)