import os
import random
from typing import List

import fasttext
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from amfbo import PROJECT_ROOT
from amfbo.utils.time_utils import cost_time

"""
    adapted from:
    
    Zhang S, Jin P, Lin Z, et al. Robust Failure Diagnosis of Microservice System through Multimodal Data[J]. 
    https://arxiv.org/abs/2302.10512
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FastTextEncoder():
    def __init__(self, modality, nodes, types, embedding_dim=100, epochs=5, random_state=1):

        self.dim = embedding_dim
        self.epochs = epochs
        self.random_state=random_state

        self.nodes = nodes
        self.types = types
        self.modality = modality

    def build_datasets(self, data_set, labels, model):
        final_data = data_set.copy()
        
        for type in self.types:
            for node in self.nodes:
                idxs = [i for i, label in enumerate(labels) if label.split('__label__')[-1] == node + str(self.types.index(type))]
                sample_count = len(idxs)
                if sample_count == 0:
                    continue
                anomaly_texts = [data_set[idx] for idx in idxs]
                loop = 0
                sample_num = 1000
                while sample_count < sample_num:
                    loop += 1
                    if loop >= 10 * sample_num:
                        break
                    chosen_text, label = anomaly_texts[random.randint(0, len(anomaly_texts) - 1)].split('\t')
                    chosen_text_splits = chosen_text.split()
                    if len(chosen_text_splits) < 1:
                        continue
                    edit_event_ids = random.sample(range(len(chosen_text_splits)), 1)
                    for event_id in edit_event_ids:
                        nearest_event = model.get_nearest_neighbors(chosen_text_splits[event_id])[0][-1]
                        chosen_text_splits[event_id] = nearest_event
                    final_data.append(
                        ' '.join(
                            chosen_text_splits) + f'\t__label__{node}{self.types.index(type)}\n')
                    sample_count += 1
        
        return final_data


    def save_to_txt(self, data_set, filename):
        tmp_dir = PROJECT_ROOT / 'tmp'
        os.makedirs(tmp_dir, exist_ok=True)
        path = str(tmp_dir / filename)
        with open(path, 'w') as f:
            for text in data_set:
                f.write(text) 

        return path

    @cost_time
    def fit(self, data_set: dict, labels: dict):
        data_set = [' '.join(events) for events in data_set]
        data_set = [f'{text}\t{labels[idx]}\n' for idx, text in enumerate(data_set)]
        train_pth = self.save_to_txt(data_set, f'{self.modality}-train.txt')

        model = fasttext.train_supervised(
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
        aug_data_set = self.build_datasets(data_set, labels, model)
        aug_train_pth = self.save_to_txt(aug_data_set, f'{self.modality}-train-aug.txt')
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
        self.event_dic = {}
        for e in model.words:
            self.event_dic[e] = model[e]
        

    def get_sentence_embedding(self, text: List[str]) -> List[float]:
        text = ' '.join(text)
        
        # senetence embedding
        length = len(self.event_dic[list(self.event_dic.keys())[0]])
        sen_emb = np.array([0] * length, 'float32')
        if text != '':
            words = list(set(text.split(' ')))
            for word in words:
                if word in self.event_dic:
                    sen_emb = sen_emb + np.array(self.event_dic[word])

        return sen_emb


    def get_words_embeddings(self, sentence: str) -> List[np.ndarray]:
        words = sentence
        embeddings = []
        for word in words:
            if word in self.event_dic:
                embeddings.append(self.event_dic[word])
            else:
                length = len(self.event_dic[list(self.event_dic.keys())[0]])
                embeddings.append(np.zeros(length, dtype='float32'))
        return embeddings


def test_fasttext_feat(X: List[List[np.ndarray]],
                    Y: List[str],
                    epochs: int = 10,
                    batch_size: int = 64,
                    lr: float = 1e-3):
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
