# import torch, numpy as np
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils import weight_norm
# from typing import List, Union
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
# # class Chomp1d(nn.Module):
# #     def __init__(self, chomp_size):
# #         super().__init__()
# #         self.chomp_size = chomp_size
# #
# #     def forward(self, x):
# #         return x[:, :, :-self.chomp_size]
# #
# # class TemporalBlock(nn.Module):
# #     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
# #         super().__init__()
# #         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
# #                                           stride=stride, padding=padding, dilation=dilation))
# #         self.chomp1 = Chomp1d(padding)
# #         self.relu1 = nn.ReLU()
# #         self.dropout1 = nn.Dropout(dropout)
# #
# #         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
# #                                           stride=stride, padding=padding, dilation=dilation))
# #         self.chomp2 = Chomp1d(padding)
# #         self.relu2 = nn.ReLU()
# #         self.dropout2 = nn.Dropout(dropout)
# #
# #         self.net = nn.Sequential(
# #             self.conv1, self.chomp1, self.relu1, self.dropout1,
# #             self.conv2, self.chomp2, self.relu2, self.dropout2
# #         )
# #         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
# #         self.relu = nn.ReLU()
# #
# #     def forward(self, x):
# #         # 添加输入验证
# #         if x.size(1) == 0:  # 检查通道维度是否为0
# #             # 返回与预期输出形状相同的零张量
# #             return torch.zeros(x.size(0), self.conv1.out_channels, x.size(2)).to(x.device)
# #
# #         out = self.net(x)
# #         res = x if self.downsample is None else self.downsample(x)
# #         return self.relu(out + res)
# #
# # class TCN(nn.Module):
# #     def __init__(self, input_size=128, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2):
# #         super().__init__()
# #         layers = []
# #         num_levels = len(num_channels)
# #         for i in range(num_levels):
# #             dilation = 2 ** i
# #             in_channels = input_size if i == 0 else num_channels[i-1]
# #             out_channels = num_channels[i]
# #             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
# #                                    dilation=dilation, padding=(kernel_size-1)*dilation,
# #                                    dropout=dropout)]
# #         self.network = nn.Sequential(*layers)
# #
# #     def forward(self, x):
# #         return self.network(x)
# #
# # class FeatTCN(nn.Module):
# #     def __init__(self, input_dim=128, hidden=64, feat_dim=32):
# #         super().__init__()
# #         self.tcn = TCN(input_size=input_dim, num_channels=[hidden])
# #         self.fc = nn.Linear(hidden, feat_dim)
# #
# #     def forward(self, x):
# #         # 输入x应该是已经padding好的tensor，形状为(batch, input_dim, seq_len)
# #         features = self.tcn(x)  # (batch, hidden, seq_len)
# #         pooled = torch.mean(features, dim=2)  # (batch, hidden)
# #         return self.fc(pooled)  # (batch, feat_dim)
#
# # ---------- 模型定义 ----------
# class FeatLSTM(nn.Module):
#     def __init__(self, input_dim=128, hidden=64, feat_dim=32):
#         super().__init__()
#         self.lstm = nn.LSTM(input_dim, hidden, batch_first=False)
#         self.fc   = nn.Linear(hidden, feat_dim)
#     def forward(self, pack):
#         _, (h_n, _) = self.lstm(pack)
#         return self.fc(h_n[-1])          # (B, 32)
#
# # ---------- 推理用包装 ----------
# class lstm_FeatExtractor:
#     def __init__(self, model, label2id):
#         self.model = model
#         self.label2id = label2id
#         self.model.eval()
#     @torch.no_grad()
#     def extract(self, seq: List[np.ndarray]) -> np.ndarray:
#         """
#         seq: list[T] of (128,) 或 整体 (T,128) 的 np.ndarray
#         返回: (32,) 向量
#         """
#         if seq.numel() == 0:  # 形状 (0,128) 也按空处理
#             return np.zeros(32, dtype=np.float32)
#
#         # ---- 2. 非空统一成 (T,128) 的 tensor ----
#         if isinstance(seq, list):
#             seq = torch.tensor(np.stack(seq), dtype=torch.float32)  # 这里现在保证不会空
#
#         seq = seq.unsqueeze(1).to(DEVICE)                     # (T,1,128)
#         lengths = torch.tensor([seq.size(0)])
#         pack = pack_padded_sequence(seq, lengths, batch_first=False, enforce_sorted=True)
#         feat = self.model(pack).squeeze(0)
#         return feat                                # (32,)
#
#
# # class tcn_FeatExtractor:
# #     def __init__(self, model, label2id):
# #         self.model = model
# #         self.label2id = label2id
# #         self.model.eval()
# #
# #     @torch.no_grad()
# #     def extract(self, seq: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
# #         """
# #         输入处理:
# #         - 如果是list: 每个元素是(128,)的np数组
# #         - 如果是np.ndarray: 形状应为(T,128)
# #         返回:
# #         - (32,)的特征向量
# #         """
# #         # 处理空序列
# #         if isinstance(seq, list):
# #             if len(seq) == 0:
# #                 return np.zeros(32, dtype=np.float32)
# #             seq = np.stack(seq)  # (T,128)
# #         elif isinstance(seq, np.ndarray):
# #             if seq.shape[0] == 0:
# #                 return np.zeros(32, dtype=np.float32)
# #
# #         # 确保输入是二维的(T,128)
# #         if seq.ndim == 1:
# #             seq = seq[np.newaxis, :]  # (1,128) -> 单帧情况
# #
# #         # 转换为torch tensor并调整维度
# #         seq_tensor = torch.tensor(seq, dtype=torch.float32)  # (T,128)
# #
# #         # 调整维度为(1, 128, T)
# #         if seq_tensor.dim() == 2:
# #             seq_tensor = seq_tensor.permute(1, 0).unsqueeze(0)  # (1,128,T)
# #
# #         # 移动到设备并计算特征
# #         seq_tensor = seq_tensor.to(DEVICE)
# #         feat = self.model(seq_tensor).squeeze(0)  # (32,)
# #
# #         return feat.cpu().numpy()
#
# # ---------- 训练函数 ----------
# def train_lstm_feat(X: List[List[np.ndarray]],       # 每条样本是 (Ti,128)
#                     Y: List[str],                    # 对应字符串标签
#                     epochs: int = 10,
#                     batch_size: int = 64,
#                     lr: float = 1e-3):
#     # 1. 标签映射
#     all_labels = sorted(set(Y))
#     label2id = {l:i for i,l in enumerate(all_labels)}
#     num_class = len(all_labels)
#
#     # 2. Dataset & collate
#     class SeqDS(Dataset):
#         def __init__(self, seqs, lbls):
#             # self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if s else torch.empty(0,128) for s in seqs]
#             self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if len(s) > 0 else torch.empty(0, 128) for s in seqs]
#             self.lbls = lbls
#         def __len__(self): return len(self.seqs)
#         def __getitem__(self, idx): return self.seqs[idx], self.lbls[idx]
#
#     def collate(batch):
#         seqs, lbs = zip(*batch)
#         # 确保所有张量在相同设备上（假设模型在 CUDA）
#         seqs = [seq.to(DEVICE) for seq in seqs]
#
#         lengths = torch.tensor([x.size(0) for x in seqs])
#         nz = [(s,i) for i,s in enumerate(seqs) if s.size(0)>0]
#         if not nz:
#             return None, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
#         nz.sort(key=lambda x: x[0].size(0), reverse=True)
#         sorted_seqs, idx = zip(*nz)
#         padded = pad_sequence(sorted_seqs, batch_first=False)
#         pack = pack_padded_sequence(padded,
#                                     lengths=torch.tensor([x.size(0) for x in sorted_seqs]),
#                                     batch_first=False, enforce_sorted=True)
#         y = torch.tensor([label2id[lbs[i]] for i in idx], dtype=torch.long)
#         return pack, y, lengths
#
#     ds = SeqDS(X, Y)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
#
#     # 3. 模型 & 训练
#     model = FeatLSTM().to(DEVICE)
#     clf_head = nn.Linear(32, num_class).to(DEVICE)
#     opt = torch.optim.AdamW(list(model.parameters())+list(clf_head.parameters()), lr=lr)
#     crit = nn.CrossEntropyLoss()
#
#     for epoch in range(epochs):
#         total, correct, loss_sum = 0, 0, 0.0
#         for pack, y, _ in dl:
#             if pack is None: continue
#             y = y.to(DEVICE)
#
#             feat = model(pack)
#             logits = clf_head(feat)
#             loss = crit(logits, y)
#             opt.zero_grad(); loss.backward(); opt.step()
#             total += y.size(0); correct += (logits.argmax(1) == y).sum().item(); loss_sum += loss.item()*y.size(0)
#         print(f'Epoch {epoch+1}/{epochs}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}')
#
#     # 4. 返回推理器
#     return lstm_FeatExtractor(model, label2id)
#
#
# # def train_tcn_feat(X: List[List[np.ndarray]], Y: List[str],
# #                    epochs: int = 10, batch_size: int = 64, lr: float = 1e-3):
# #     # 1. 标签映射
# #     all_labels = sorted(set(Y))
# #     label2id = {l: i for i, l in enumerate(all_labels)}
# #     num_class = len(all_labels)
# #
# #     # 2. Dataset
# #     class SeqDS(Dataset):
# #         def __init__(self, seqs, lbls):
# #             self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if len(s) > 0
# #                          else torch.empty(0, 128) for s in seqs]
# #             self.lbls = lbls
# #
# #         def __len__(self):
# #             return len(self.seqs)
# #
# #         def __getitem__(self, idx):
# #             return self.seqs[idx], self.lbls[idx]
# #
# #     def collate_fn(batch):
# #         seqs, lbs = zip(*batch)
# #
# #         # 过滤空序列
# #         non_empty = [(s, l) for s, l in zip(seqs, lbs) if s.size(0) > 0]
# #         if not non_empty:
# #             return None, None, None
# #
# #         seqs, lbs = zip(*non_empty)
# #         lengths = torch.tensor([s.size(0) for s in seqs])
# #
# #         # 按长度排序(从长到短)
# #         sorted_indices = torch.argsort(lengths, descending=True)
# #         sorted_seqs = [seqs[i] for i in sorted_indices]
# #         y = torch.tensor([label2id[lbs[i]] for i in sorted_indices], dtype=torch.long)
# #
# #         # 填充并转置为(batch, input_dim, seq_len)
# #         padded = pad_sequence(sorted_seqs, batch_first=False)  # (seq_len, batch, input_dim)
# #         padded = padded.permute(1, 2, 0)  # (batch, input_dim, seq_len)
# #
# #         return padded.to(DEVICE), y.to(DEVICE), lengths[sorted_indices]
# #
# #     ds = SeqDS(X, Y)
# #     dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
# #                     collate_fn=collate_fn, drop_last=False)
# #
# #     # 3. 模型 & 训练
# #     model = FeatTCN().to(DEVICE)
# #     clf_head = nn.Linear(32, num_class).to(DEVICE)
# #     opt = torch.optim.AdamW(list(model.parameters()) + list(clf_head.parameters()), lr=lr)
# #     crit = nn.CrossEntropyLoss()
# #
# #     for epoch in range(epochs):
# #         total, correct, loss_sum = 0, 0, 0.0
# #         for padded, y, _ in dl:
# #             if padded is None:
# #                 continue
# #
# #             feat = model(padded)
# #             logits = clf_head(feat)
# #             loss = crit(logits, y)
# #
# #             opt.zero_grad()
# #             loss.backward()
# #             opt.step()
# #
# #             total += y.size(0)
# #             correct += (logits.argmax(1) == y).sum().item()
# #             loss_sum += loss.item() * y.size(0)
# #
# #         print(f'Epoch {epoch + 1}/{epochs}  loss={loss_sum / total:.4f}  acc={correct / total:.4f}')
# #
# #     # 4. 返回推理器
# #     return tcn_FeatExtractor(model, label2id)
#
#
# def get_LSTM_model(X_list, Y_list, epochs=30, batch_size=64):
#
#     return train_lstm_feat(X_list, Y_list, epochs, batch_size)



import torch, numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 模型定义 ----------
class FeatLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden=64, feat_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=False)
        self.fc   = nn.Linear(hidden, feat_dim)
    def forward(self, pack):
        _, (h_n, _) = self.lstm(pack)
        return self.fc(h_n[-1])          # (B, 32)
# class FeatLSTM(nn.Module):
#     def __init__(self, input_dim=128, hidden=64, feat_dim=32):
#         super().__init__()
#         # 双向LSTM，bidirectional=True
#         self.lstm = nn.LSTM(input_dim, hidden, batch_first=False, bidirectional=True)
#         # 因为双向LSTM的输出维度是 hidden*2
#         self.fc = nn.Linear(hidden * 2, feat_dim)
#
#     def forward(self, pack):
#         # pack shape: (seq_len, batch, input_dim)
#         _, (h_n, _) = self.lstm(pack)
#         # h_n shape: (num_layers * num_directions, batch, hidden_size)
#
#         # 合并双向LSTM的最后隐藏状态
#         # 前向的最后一个隐藏状态和后向的第一个隐藏状态
#         forward_hidden = h_n[-2]  # 前向的最后一个隐藏状态
#         backward_hidden = h_n[-1]  # 后向的第一个隐藏状态
#
#         # 拼接两个方向的隐藏状态
#         combined = torch.cat((forward_hidden, backward_hidden), dim=1)
#
#         return self.fc(combined)  # (B, 32)


# ---------- 推理用包装 ----------
class FeatExtractor:
    def __init__(self, model, label2id):
        self.model = model
        self.label2id = label2id
        self.model.eval()
    @torch.no_grad()
    def extract(self, seq: List[np.ndarray]) -> np.ndarray:
        """
        seq: list[T] of (128,) 或 整体 (T,128) 的 np.ndarray
        返回: (32,) 向量
        """
        if seq.numel() == 0:  # 形状 (0,128) 也按空处理
            return np.zeros(32, dtype=np.float32)

        # ---- 2. 非空统一成 (T,128) 的 tensor ----
        if isinstance(seq, list):
            seq = torch.tensor(np.stack(seq), dtype=torch.float32)  # 这里现在保证不会空

        seq = seq.unsqueeze(1).to(DEVICE)                     # (T,1,128)
        lengths = torch.tensor([seq.size(0)])
        pack = pack_padded_sequence(seq, lengths, batch_first=False, enforce_sorted=True)
        feat = self.model(pack).squeeze(0)
        return feat                                # (32,)

# ---------- 训练函数 ----------
def train_lstm_feat(X: List[List[np.ndarray]],       # 每条样本是 (Ti,128)
                    Y: List[str],                    # 对应字符串标签
                    epochs: int = 50,
                    batch_size: int = 64,
                    lr: float = 1e-3):
    # 1. 标签映射
    all_labels = sorted(set(Y))
    label2id = {l:i for i,l in enumerate(all_labels)}
    num_class = len(all_labels)

    # 2. Dataset & collate
    class SeqDS(Dataset):
        def __init__(self, seqs, lbls):
            # self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if s else torch.empty(0,128) for s in seqs]
            self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if len(s) > 0 else torch.empty(0, 128) for s in seqs]
            self.lbls = lbls
        def __len__(self): return len(self.seqs)
        def __getitem__(self, idx): return self.seqs[idx], self.lbls[idx]

    def collate(batch):
        seqs, lbs = zip(*batch)
        # 确保所有张量在相同设备上（假设模型在 CUDA）
        seqs = [seq.to(DEVICE) for seq in seqs]

        lengths = torch.tensor([x.size(0) for x in seqs])
        nz = [(s,i) for i,s in enumerate(seqs) if s.size(0)>0]
        if not nz:
            return None, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        nz.sort(key=lambda x: x[0].size(0), reverse=True)
        sorted_seqs, idx = zip(*nz)
        padded = pad_sequence(sorted_seqs, batch_first=False)
        pack = pack_padded_sequence(padded,
                                    lengths=torch.tensor([x.size(0) for x in sorted_seqs]),
                                    batch_first=False, enforce_sorted=True)
        y = torch.tensor([label2id[lbs[i]] for i in idx], dtype=torch.long)
        return pack, y, lengths

    ds = SeqDS(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)

    # 3. 模型 & 训练
    model = FeatLSTM().to(DEVICE)
    clf_head = nn.Linear(32, num_class).to(DEVICE)
    opt = torch.optim.AdamW(list(model.parameters())+list(clf_head.parameters()), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for pack, y, _ in dl:
            if pack is None: continue
            y = y.to(DEVICE)
            feat = model(pack)
            logits = clf_head(feat)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += y.size(0); correct += (logits.argmax(1) == y).sum().item(); loss_sum += loss.item()*y.size(0)
        print(f'Epoch {epoch+1}/{epochs}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}')

    # 4. 返回推理器
    return FeatExtractor(model, label2id)


def get_LSTM_model(X_list, Y_list, epochs=30, batch_size=64):

    return train_lstm_feat(X_list, Y_list, epochs, batch_size)
