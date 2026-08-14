import json
import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

from amfbo import PROJECT_ROOT
from amfbo.config.experiment_config import Config
from amfbo.core.augmentation import *
from amfbo.core.models.temporal_cnn_lstm import main_training_function
from amfbo.core.multimodal_data import MultiModalDataset
from amfbo.pipeline.events.fasttext_encoder import (FastTextEncoder,
                                                    test_fasttext_feat)
from amfbo.pipeline.events.lstm_feature import get_LSTM_model
from amfbo.utils import io_utils
from amfbo.utils.dataset_io import save_dataset
from amfbo.utils.logger import get_logger


class EventPipeline():

    def __init__(self, config: Config, logger):
        self.config = config
        self.logger = logger
        self.dataset = config.dataset

    def process(self, reconstruct=False, exp_name=None):
        data_dir = PROJECT_ROOT / 'data' / self.dataset
        self.data_path = str(data_dir)
        label_path = str(data_dir / 'label.csv')
        metric_path = str(data_dir / 'raw' / 'metrics.json')
        trace_path = str(data_dir / 'raw' / 'traces.json')
        log_path = str(data_dir / 'raw' / 'logs.json')
        edge_path = str(data_dir / 'raw' / 'edges.json')
        node_path = str(data_dir / 'raw' / 'nodes.json')

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)
        self.labels['index'] = self.labels['index'].astype(str)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)
        with open(edge_path, 'r', encoding='utf8') as fp:
            self.edges = json.load(fp)
        with open(node_path, 'r', encoding='utf8') as fp:
            self.nodes = json.load(fp)

        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist()

        if reconstruct:
            self.build_embedding()

        self.build_dataset(exp_name)

    def build_embedding(self):
        self.logger.info(f"Build embedding for raw events")

        data_map = {'log': self.logs, 'trace': self.traces, 'metric': self.metrics}

        for key, data in data_map.items():
            all_nodes = list({item for sublist in self.nodes.values() for item in sublist})

            encoder = FastTextEncoder(key, all_nodes, self.types, embedding_dim=self.config.alert_embedding_dim, epochs=5)

            train_idxs = self.labels[self.labels['data_type']=='train']['index'].values.tolist()
            train_ins_labels = self.labels[self.labels['data_type']=='train']['instance'].values.tolist()
            train_type_labels = self.labels[self.labels['data_type']=='train']['anomaly_type'].values.tolist()

            docs = []
            labels = []
            for i, idx in enumerate(train_idxs):
                nodes = self.nodes[str(idx)]
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                            # ['logservice1&dbservice1&http://0.0.0.4:9388/db_login_methods&PD']
                            # ['logservice2&redisservice2&http://0.0.0.2:9387/get_value_from_redis&ERROR', 'mobservice1&redisservice2&http://0.0.0.2:9387/get_value_from_redis&ERROR']
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    docs.append(doc)
                    if node == train_ins_labels[i]:
                        labels.append(f'__label__{node}{self.types.index(train_type_labels[i])}')
                    else:
                        labels.append(f'__label__{node}0')

            encoder.fit(docs, labels)

            word_embeddings = []
            for i, idx in enumerate(train_idxs):
                nodes = self.nodes[str(idx)]
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]

                    word_embedding = encoder.get_words_embeddings(doc)
                    word_embedding = torch.FloatTensor(np.array(word_embedding))
                    word_embeddings.append(word_embedding)

            model_lstm = get_LSTM_model(word_embeddings, labels, epochs=100, batch_size=64)

            # build embedding
            embs = {}
            embs_word = {}
            for idx in self.labels['index']:
                # group by instance
                graph_embs = []
                word_embs = []
                for node in nodes:
                    if key == 'trace':
                        if self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif not self.config.trace_op and self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]

                    word_embedding = encoder.get_words_embeddings(doc)
                    word_embedding = torch.FloatTensor(np.array(word_embedding))
                    feat = model_lstm.extract(word_embedding)
                    word_embs.append(feat)

                    emb = encoder.get_sentence_embedding(doc)
                    graph_embs.append(emb)

                embs_word[idx] = word_embs
                embs[idx]=graph_embs
            print("success")

            # print(len(embs))
            # print(len(embs_word))
            tmp_pth = PROJECT_ROOT / 'data' / self.dataset / 'tmp'
            os.makedirs(tmp_pth, exist_ok=True)
            io_utils.save_pkl(str(tmp_pth / f"{key}.pkl"), embs)
            io_utils.save_pkl(str(tmp_pth / f"{key}_word.pkl"), embs_word)


    def build_dataset(self, exp_name):
        self.logger.info(f"Build dataset for training")
        tmp_pth = PROJECT_ROOT / 'data' / self.dataset / 'tmp'
        metric_embs = io_utils.load_pkl(str(tmp_pth / 'metric.pkl'))
        trace_embs = io_utils.load_pkl(str(tmp_pth / 'trace.pkl'))
        log_embs = io_utils.load_pkl(str(tmp_pth / 'log.pkl'))
        metric_embs_word = io_utils.load_pkl(str(tmp_pth / 'metric_word.pkl'))
        trace_embs_word = io_utils.load_pkl(str(tmp_pth / 'trace_word.pkl'))
        log_embs_word = io_utils.load_pkl(str(tmp_pth / 'log_word.pkl'))

        label_dict = {}

        all_nodes = list({item for sublist in self.nodes.values() for item in sublist})
        all_nodes.sort()
        print(all_nodes)

        label_dict['instance'], node2idx, idx2root = self.get_root_labels(all_nodes)

        label_dict['anomaly_type'], ft2idx, idx2ft = self.get_type_labels(self.labels['anomaly_type'].values.tolist())
        train_data, test_data = MultiModalDataset(), MultiModalDataset()
        for _, row in self.labels.iterrows():
            index = str(row['index'])
            data_type = row['data_type']
            metric_Xs, trace_Xs, log_Xs, metric_Xs_word, trace_Xs_word, log_Xs_word = metric_embs[index], trace_embs[index], log_embs[index], metric_embs_word[index], trace_embs_word[index], log_embs_word[index]

            global_root_id = node2idx[row['instance']]
            failure_type_id = ft2idx[row['anomaly_type']]
            nodes = self.nodes[index]
            edges = self.edges[index]
            
            if data_type == 'train':
                train_data.add_data(
                    metric_Xs=metric_Xs, 
                    trace_Xs=trace_Xs, 
                    log_Xs=log_Xs,
                    metric_Xs_word=metric_Xs_word,
                    trace_Xs_word=trace_Xs_word,
                    log_Xs_word=log_Xs_word,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=row['instance'],
                    nodes=nodes,
                    edges=edges)
            else:
                test_data.add_data(
                    metric_Xs=metric_Xs,
                    trace_Xs=trace_Xs,
                    log_Xs=log_Xs,
                    metric_Xs_word=metric_Xs_word,
                    trace_Xs_word=trace_Xs_word,
                    log_Xs_word=log_Xs_word,
                    global_root_id=global_root_id,
                    failure_type_id=failure_type_id,
                    local_root=row['instance'],
                    nodes=nodes,
                    edges=edges)
        aug_data = []
        if self.config.aug_times > 0:
            # for time in range(self.config.aug_times):
            for time in range(6):
                for (graph, labels) in train_data:
                    root = graph.ndata['root'].tolist().index(1)
                    aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)

                    aug_data.append((aug_graph, labels))

        for (graph, labels) in train_data:
            aug_data.append((graph, labels))

        train_dl = DataLoader(aug_data, batch_size=32, shuffle=True, collate_fn=self.collate)
        save_dataset(train_dl, 'aug_data', exp_name)

        test_dl = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=self.collate)
        save_dataset(test_dl, 'test_data', exp_name)

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def get_root_labels(self, nodes):
        labels2idx = {node: idx for idx, node in enumerate(nodes)}
        idx2label = {idx: label for idx, label in enumerate(nodes)}
        labels = np.array(self.labels['instance'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label

    def get_type_labels(self, types):
        meta_labels = sorted(list(set(types)))
        labels2idx = {label: idx for idx, label in enumerate(meta_labels)}
        idx2label = {idx: label for idx, label in enumerate(meta_labels)}
        labels = np.array(self.labels['anomaly_type'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label

