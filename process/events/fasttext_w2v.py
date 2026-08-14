import os
import random
import fasttext
import numpy as np
from helper.time_util import cost_time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List

"""
    adapted from:
    
    Zhang S, Jin P, Lin Z, et al. Robust Failure Diagnosis of Microservice System through Multimodal Data[J]. 
    https://arxiv.org/abs/2302.10512
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 这个文件定义了一个类 FastTextEncoder，用于使用 FastText 模型生成文本嵌入向量。      这个类特别适用于处理多模态数据中的文本特征，例如日志、追踪等。
class FastTextEncoder():
    def __init__(self, modality, nodes, types, embedding_dim=100, epochs=5, random_state=1):
        '''
            modality：当前处理的数据模态（如 metric、trace、log）。
            nodes：所有节点的列表。
            types：所有异常类型的列表。
            embedding_dim：嵌入向量的维度，默认为 100。
            epochs：训练 FastText 模型的轮数，默认为 5。
            random_state：随机种子，用于确保结果的可重复性。
        '''
        self.dim = embedding_dim
        self.epochs = epochs
        self.random_state=random_state

        self.nodes = nodes
        self.types = types
        self.modality = modality

    def build_datasets(self, data_set, labels, model):
        final_data = data_set.copy()    #　创建一个副本，用于存储增强后的数据集。
        
        for type in self.types: # 遍历所有异常类型和节点，找到对应类型的索引。
            for node in self.nodes:
                # 统计当前异常类型的数量
                idxs = [i for i, label in enumerate(labels) if label.split('__label__')[-1] == node + str(self.types.index(type))]
                sample_count = len(idxs)
                if sample_count == 0:   # 如果当前类型的样本数量为 0，则跳过。
                    continue
                anomaly_texts = [data_set[idx] for idx in idxs] # 获取当前类型的异常文本句子。
                loop = 0
                sample_num = 1000
                while sample_count < sample_num:    # 设置一个循环，最多尝试 10 倍的样本数量次，以避免无限循环。
                    loop += 1
                    if loop >= 10 * sample_num:
                        break
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')   # 随机选择一个异常文本，将其拆分为单词。
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < 1: # 如果单词数量小于 1，则跳过。
                        continue
                    edit_event_ids = random.sample(range(len(chosen_text_splits)), 1)   # 随机选中一个单词的下标，组成一个列表
                    for event_id in edit_event_ids: # 随机选择一个单词，用其最近邻单词替换。
                        nearest_event = model.get_nearest_neighbors(chosen_text_splits[event_id])[0][-1]
                        # chosen_text_splits[event_id] → 获取目标文本     [0] → 取第一个（最相似）结果       [-1] → 取结果中的最后一个元素（通常是文本本身）
                        chosen_text_splits[event_id] = nearest_event
                    final_data.append(
                        ' '.join(
                            chosen_text_splits) + f'\t__label__{node}{self.types.index(type)}\n')   # 将增强后的文本添加到数据集中。
                    sample_count += 1
        
        return final_data


    def save_to_txt(self, data_set, filename):
        os.makedirs('./tmp', exist_ok=True)
        path = f'./tmp/{filename}'
        with open(path, 'w') as f:
            for text in data_set:
                f.write(text) 

        return path

    @cost_time
    def fit(self, data_set: dict, labels: dict):    # data_set：原始数据集。       labels：标签列表。
        # 将属于10个节点的各自的不同模态的事件用' '空格连接成一句话
        data_set = [' '.join(events) for events in data_set]    # 将数据集中的每个事件列表转换为字符串，并与对应的标签组合。
        data_set = [f'{text}\t{labels[idx]}\n' for idx, text in enumerate(data_set)]
        # 得到的结果data_set为一个列表，'\t__label__consul-00\n'其中\t和\n真实存在
        train_pth = self.save_to_txt(data_set, f'./{self.modality}-train.txt')    # 将数据集保存到文件。

        model = fasttext.train_supervised(  # 使用 FastText 训练一个监督模型。
            train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )
        # model = fasttext.train_unsupervised(
        #     train_pth,
        #     dim=self.dim,
        #     minCount=1, 
        #     minn=0, maxn=0, epoch=self.epochs
        # )

        # aug
        aug_data_set = self.build_datasets(data_set, labels, model)     # 对数据集进行增强，并重新训练模型。
        aug_train_pth = self.save_to_txt(aug_data_set, f'./{self.modality}-train-aug.txt')
        model = fasttext.train_supervised(
            aug_train_pth,
            dim=self.dim,
            minCount=1, 
            minn=0, maxn=0, epoch=self.epochs
        )
        # model = fasttext.train_unsupervised(
        #     aug_train_pth,
        #     dim=self.dim,
        #     minCount=1, 
        #     minn=0, maxn=0, epoch=self.epochs
        # )

        # event embedding
        self.event_dic = {} # 生成事件嵌入字典。
        for e in model.words:
            self.event_dic[e] = model[e]
        

    def get_sentence_embedding(self, text: List[str]) -> List[float]:
        # text：输入文本，是一个单词列表。
        text = ' '.join(text)   # 将单词列表转换为字符串句子。
        
        # senetence embedding
        length = len(self.event_dic[list(self.event_dic.keys())[0]])    # 初始化嵌入向量为零向量。
        sen_emb = np.array([0] * length, 'float32')
        if text != '':       # 如果文本不为空，将文本拆分为单词，并累加每个单词的嵌入向量。
            words = list(set(text.split(' ')))
            for word in words:
                if word in self.event_dic:
                    sen_emb = sen_emb + np.array(self.event_dic[word])

        return sen_emb


    def get_words_embeddings(self, sentence: str) -> List[np.ndarray]:
        """
        获取句子中所有单词的嵌入向量列表
        :param sentence: 输入句子字符串
        :return: 单词嵌入向量列表
        """
        words = sentence
        embeddings = []
        for word in words:
            if word in self.event_dic:
                embeddings.append(self.event_dic[word])
            else:
                # 处理未登录词
                length = len(self.event_dic[list(self.event_dic.keys())[0]])
                embeddings.append(np.zeros(length, dtype='float32'))
        return embeddings


def test_fasttext_feat(X: List[List[np.ndarray]],       # 每条样本是 (Ti,128)
                    Y: List[str],                    # 对应字符串标签
                    epochs: int = 10,
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
            # self.seqs = [torch.tensor(np.stack(s), dtype=torch.float32) if len(s) > 0 else torch.empty(0, 128) for s in seqs]
            self.seqs = seqs
            self.lbls = lbls
        def __len__(self): return len(self.seqs)
        def __getitem__(self, idx): return self.seqs[idx], self.lbls[idx]

    # def collate(batch):
    #     seqs, lbs = zip(*batch)
    #     # 确保所有张量在相同设备上（假设模型在 CUDA）
    #     seqs = [seq.to(DEVICE) for seq in seqs]
    #
    #     lengths = torch.tensor([x.size(0) for x in seqs])
    #     nz = [(s,i) for i,s in enumerate(seqs) if s.size(0)>0]
    #     if not nz:
    #         return None, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    #     nz.sort(key=lambda x: x[0].size(0), reverse=True)
    #     sorted_seqs, idx = zip(*nz)
    #     padded = pad_sequence(sorted_seqs, batch_first=False)
    #     pack = pack_padded_sequence(padded,
    #                                 lengths=torch.tensor([x.size(0) for x in sorted_seqs]),
    #                                 batch_first=False, enforce_sorted=True)
    #     y = torch.tensor([label2id[lbs[i]] for i in idx], dtype=torch.long)
    #     return pack, y, lengths

    ds = SeqDS(X, Y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)   # collate_fn=collate,

    # 3. 模型 & 训练
    aa = nn.Sequential(
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,num_class)
    )
    clf_head = aa.to(DEVICE)
    opt = torch.optim.AdamW(clf_head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for pack, y, _ in dl:
            if pack is None: continue
            y = y.to(DEVICE)
            logits = clf_head(pack)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += y.size(0); correct += (logits.argmax(1) == y).sum().item(); loss_sum += loss.item()*y.size(0)
        print(f'Epoch {epoch+1}/{epochs}  loss={loss_sum/total:.4f}  acc={correct/total:.4f}')
