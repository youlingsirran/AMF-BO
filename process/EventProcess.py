import os
from torch.utils.data import DataLoader, random_split
from config.exp_config import Config
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
import json, random
import pandas as pd
import numpy as np
from process.events.fasttext_w2v import FastTextEncoder, test_fasttext_feat
from process.events.test import get_LSTM_model
from core.model.tcn_lstm import main_training_function
from core.aug import *
from helper.logger import get_logger
from MyTest import save_dataset
# EventProcess 类用于处理事件数据，包括加载原始数据、构建嵌入向量、生成数据集以及进行数据增强等操作。
class EventProcess():

    def __init__(self, config: Config, logger):
        self.config = config    # Config 类的实例，包含实验配置参数。
        self.logger = logger
        self.dataset = config.dataset

    def process(self, reconstruct=False):
        self.data_path = f"data/{self.dataset}" # 数据集的根目录路径。
        label_path = f"data/{self.dataset}/label.csv"   # 标签文件的路径。
        metric_path = f"data/{self.dataset}/raw/metrics.json"   # 度量数据的路径。
        trace_path = f"data/{self.dataset}/raw/traces.json" # 追踪数据的路径。
        log_path = f"data/{self.dataset}/raw/logs.json" # 日志数据的路径。
        edge_path = f"data/{self.dataset}/raw/edges.json"   # 边数据的路径。
        node_path = f"data/{self.dataset}/raw/nodes.json"   # 节点数据的路径。

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)   # 加载标签文件，将 index 列转换为字符串类型。
        self.labels['index'] = self.labels['index'].astype(str)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)    # 加载度量数据。
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp) # 加载追踪数据。
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)   # 加载日志数据。
        with open(edge_path, 'r', encoding='utf8') as fp:
            self.edges = json.load(fp)  # 加载边数据。
        with open(node_path, 'r', encoding='utf8') as fp:
            self.nodes = json.load(fp)  # 加载节点数据。

        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist() # self.types：获取所有异常类型，并将其与 'normal' 类型一起存储为列表。

        if reconstruct: # 如果 reconstruct 为 True，则调用 build_embedding 方法重新构建嵌入
            self.build_embedding()

        self.build_dataset()

    def build_embedding(self):  # 构建事件的嵌入（embedding）。   构建事件的嵌入向量。
        self.logger.info(f"Build embedding for raw events") # 记录日志，表示开始构建嵌入。
        # metric event: (instance, metric_name, abnormal type)
        # trace event: (src, dst, op, error_type)
        # log event: (instance, eventId)

        # data_map = {'metric': self.metrics, 'trace': self.traces, 'log': self.logs}  # data_map：将不同模态的数据（度量、追踪、日志）映射到对应的变量。
        # data_map = {'trace': self.traces}
        data_map = {'log': self.logs, 'trace': self.traces, 'metric': self.metrics}
        # print(len(self.traces))   # 1099 条

        for key, data in data_map.items():
            # key 表示模态名称[metric、trace、log]之一，data表示对应的读取的文本类型的事件的文件，是用 json.load()读取的DataFram格式的文件
            all_nodes = list({item for sublist in self.nodes.values() for item in sublist}) # 获取所有节点的唯一列表。
            # all_nodes = ['webservice2', 'logservice1', 'dbservice1', 'mobservice2', 'webservice1', 'logservice2', 'redisservice1', 'redisservice2', 'dbservice2', 'mobservice1']

            # 初始化 FastText 编码器： FastTextEncoder：用于构建嵌入的工具。  key：当前处理的数据模态（metric、trace、log）。 embedding_dim：嵌入的维度128。 epochs：训练嵌入的轮数。
            encoder = FastTextEncoder(key, all_nodes, self.types, embedding_dim=self.config.alert_embedding_dim, epochs=5)

            # 获取训练数据的索引、实例标签和异常类型标签。
            train_idxs = self.labels[self.labels['data_type']=='train']['index'].values.tolist()    # 获得训练数据集的索引、故障实例、异常类型
            # train_idxs是保存了data_type为train的所有数据的index列的数组
            train_ins_labels = self.labels[self.labels['data_type']=='train']['instance'].values.tolist()
            train_type_labels = self.labels[self.labels['data_type']=='train']['anomaly_type'].values.tolist()

            docs = []   #　初始化 docs 和 labels 列表，用于存储训练数据的文档和标签。
            labels = []
            for i, idx in enumerate(train_idxs):         # 遍历训练数据的索引，根据当前节点和数据模态构建文档和标签。
                nodes = self.nodes[str(idx)]    # 获取当前索引对应的节点列表，即当前故障共涉及到多少个微服务节点
                for node in nodes:  # 遍历当前索引对应的节点列表。
                    if key == 'trace':  # key：当前处理的数据模态（metric、trace、log）  data：一个字典，键为索引，值为对应模态的数据。  data[str(idx)]：当前索引对应的模态数据。
                        if self.config.trace_op and self.config.trace_ab_type:  # 如果 trace_op 和 trace_ab_type 都为 True，则文档为 e 的所有元素用 & 连接
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]     # doc：根据数据模态和配置生成的文档。
                            # 会将与当前节点相关的所有trace数据分别用 & 连接起来后，放入一个列表里面
                            # ['logservice1&dbservice1&http://0.0.0.4:9388/db_login_methods&PD']
                            # ['logservice2&redisservice2&http://0.0.0.2:9387/get_value_from_redis&ERROR', 'mobservice1&redisservice2&http://0.0.0.2:9387/get_value_from_redis&ERROR']
                        elif not self.config.trace_op and self.config.trace_ab_type:  # 如果 trace_op 为 False 且 trace_ab_type 为 True，则文档为 e 的前两个元素用 & 连接，跳过第三个元素。
                            doc=['&'.join([item for i, item in enumerate(e) if i != 2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        elif self.config.trace_op and not self.config.trace_ab_type:    # 如果 trace_op 为 True 且 trace_ab_type 为 False，则文档为 e 的前三个元素用 & 连接，跳过第四个元素
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                        else:   # 如果 trace_op 和 trace_ab_type 都为 False，则文档为 e 的前两个元素用 & 连接。
                            doc=['&'.join(e[:2]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    elif key == 'metric':
                        if self.config.metric_direction:    # 如果 metric_direction 为 True，则文档为 e 的所有元素用 & 连接。
                            doc=['&'.join(e) for e in data[str(idx)] if (node in e[0])]
                        else:   # 如果 metric_direction 为 False，则文档为 e 的前三个元素用 & 连接，跳过第四个元素。
                            doc=['&'.join([item for i, item in enumerate(e) if i != 3]) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]   # 文档为 e 的所有元素用 & 连接。
                    docs.append(doc)    # docs：存储所有生成的文档。
                    if node == train_ins_labels[i]:     # self.types.index(train_type_labels[i])：获取异常类型在 self.types 列表中的索引。
                        labels.append(f'__label__{node}{self.types.index(train_type_labels[i])}')   # 如果当前节点是实例标签，则标签为 __label__{node}{type_index}。
                    else:
                        labels.append(f'__label__{node}0')  # 如果当前节点不是实例标签，则标签为 __label__{node}0。

            encoder.fit(docs, labels)       # 使用构建的文档和标签训练 FastText 编码器。
            # print(len(docs))    # 1600 条，表示160个train数据，每个train数据有10个instance，所以有10个标签，共1600个标签
            # print(len(labels))  # 1600 条

            word_embeddings = []
            for i, idx in enumerate(train_idxs):    # 遍历所有数据的索引，为每个节点构建嵌入。
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
                    # print(doc)
                    # print(word_embedding)
            # model_lstm = main_training_function(word_embeddings,labels)
            model_lstm = get_LSTM_model(word_embeddings, labels, epochs=100, batch_size=64)

            # build embedding
            embs = {}
            embs_word = {}
            for idx in self.labels['index']:    # 遍历所有数据的索引，为每个节点构建嵌入。
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
                    graph_embs.append(emb)  # graph_embs：存储当前图的所有节点嵌入。

                embs_word[idx] = word_embs
                embs[idx]=graph_embs    # embs：存储所有图的嵌入。embs为一个字典，值为序号，键为该序号对应的记录的某一模态的向量表示
            print("success")

            # print(len(embs))
            # print(len(embs_word))
            tmp_pth = f'data/{self.dataset}/tmp/'   # 保存嵌入
            if not os.path.isdir(tmp_pth):
                os.system(f"mkdir -p {tmp_pth}")
            io_util.save_pkl(f"{tmp_pth}/{key}.pkl", embs)
            io_util.save_pkl(f"{tmp_pth}/{key}_word.pkl", embs_word)


    def build_dataset(self):    # 构建训练数据集、增强数据集和测试数据集。
        self.logger.info(f"Build dataset for training") #　记录日志，表示开始构建数据集。
        metric_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/metric.pkl")   #　加载之前保存的度量、追踪和日志嵌入。
        trace_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/trace.pkl") # io_util.load_pkl：从指定路径加载 .pkl 文件。
        log_embs = io_util.load_pkl(f"data/{self.dataset}/tmp/log.pkl")
        metric_embs_word = io_util.load_pkl(f"data/{self.dataset}/tmp/metric_word.pkl")
        trace_embs_word = io_util.load_pkl(f"data/{self.dataset}/tmp/trace_word.pkl")
        log_embs_word = io_util.load_pkl(f"data/{self.dataset}/tmp/log_word.pkl")

        label_dict = {} # label_dict：用于存储不同类型的标签。

        all_nodes = list({item for sublist in self.nodes.values() for item in sublist})
        all_nodes.sort() # all_nodes：获取所有节点的唯一列表，并对其进行排序。
        print(all_nodes)
        # all_nodes = ['dbservice1', 'dbservice2', 'logservice1', 'logservice2', 'mobservice1', 'mobservice2', 'redisservice1', 'redisservice2', 'webservice1', 'webservice2']

        label_dict['instance'], node2idx, idx2root = self.get_root_labels(all_nodes)
        # 获取实例标签的索引映射和反向映射。
            # node2idx：将节点名称映射到索引。
            # idx2root：将索引映射到节点名称。
            # label_dict['instance']    将标签数据中的节点名称转换为对应的索引。

        label_dict['anomaly_type'], ft2idx, idx2ft = self.get_type_labels(self.labels['anomaly_type'].values.tolist())
        # print("AAAA",label_dict['anomaly_type'])    # [2 3 3 ... 2 3 2]
        # print("AAAABBBB",ft2idx)    # {'[access permission denied exception]': 0, '[file moving program]': 1, '[login failure]': 2, '[memory_anomalies]': 3, '[normal memory freed label]': 4}
        # print("AAAACCCC",idx2ft)    # {0: '[access permission denied exception]', 1: '[file moving program]', 2: '[login failure]', 3: '[memory_anomalies]', 4: '[normal memory freed label]'}

        # print(label_dict)   # {'instance': array([4, 5, 2, ..., 4, 0, 5]), 'anomaly_type': array([2, 3, 3, ..., 2, 3, 2])}

        train_data, test_data = MultiModalDataSet(), MultiModalDataSet()    # 创建训练数据集和测试数据集。
        # train_data.data包含的是一个列表，里面的每一个元组元素表示label.csv文件里面的一行，每一个元组元素包括2个元素，对应图graph，和根因，异常类型（root, anormal）
        for _, row in self.labels.iterrows():   # 遍历标签数据，构建数据集
            index = str(row['index'])   # 当前数据的索引。
            data_type = row['data_type']    # 数据类型（train 或 test）。
            # 分别获取度量、追踪和日志的嵌入。
            metric_Xs, trace_Xs, log_Xs, metric_Xs_word, trace_Xs_word, log_Xs_word = metric_embs[index], trace_embs[index], log_embs[index], metric_embs_word[index], trace_embs_word[index], log_embs_word[index]

            # 数值labels
            global_root_id = node2idx[row['instance']] #  全局根因标签
            failure_type_id = ft2idx[row['anomaly_type']]   # 故障类型标签。
            # topo  当前数据的节点和边信息。
            nodes = self.nodes[index]
            edges = self.edges[index]
            
            if data_type == 'train':    # 根据数据类型（train 或 test），将数据添加到相应的数据集中。
                # 在这里，每个训练数据表示为一个图，图中节点有指标、追踪和日志3个属性和一个根因标签
                train_data.add_data(
                    metric_Xs=metric_Xs, 
                    trace_Xs=trace_Xs, 
                    log_Xs=log_Xs,
                    metric_Xs_word=metric_Xs_word,
                    trace_Xs_word=trace_Xs_word,
                    log_Xs_word=log_Xs_word,
                    global_root_id=global_root_id,  # global_root_id，表示当前行对应的节点名称在 node2idx 中的映射id
                    failure_type_id=failure_type_id, # 当前行的异常类型的映射标签
                    local_root=row['instance'],     # 当前行的异常节点的名称
                    nodes=nodes, # 那 10 个节点名称的列表
                    edges=edges)    # 边的连接，每行异常的边的连接情况并不相同
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

        # print(len(train_data.data))     # 160
        # print(len(test_data.data))      # 939
        # print(train_data.data[1])     # 打印训练集中的某个元素
        # # (Graph(num_nodes=10, num_edges=39,
        # #       ndata_schemes={'metric': Scheme(shape=(128,), dtype=torch.float32), 'trace': Scheme(shape=(128,), dtype=torch.float32), 'log': Scheme(shape=(128,), dtype=torch.float32), 'root': Scheme(shape=(), dtype=torch.int64)}
        # #       edata_schemes={}), (5, 3))          最后的（5，3）对应异常节点和根因

        print("到此正确划分了训练数据集和测试数据集train_data和test_data")

        print("下面开始利用train_data构建增强数据集aug_data")

        aug_data = []   # 增强数据集
        if self.config.aug_times > 0:   # self.config.aug_times：增强次数。          self.aug_times = 10 # 数据增强的次数，默认为 10。
            # for time in range(self.config.aug_times):
            for time in range(6):
                for (graph, labels) in train_data:  # 通过丢弃节点进行数据增强。
                    root = graph.ndata['root'].tolist().index(1)    # 找到根节点。
                    aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)    # aug_graph：增强后的图   self.aug_percent = 0.2  # 数据增强的百分比，默认为 0.2。
                    # if time == 0:
                    #     print(aug_graph)
                    # # Graph(num_nodes=8, num_edges=13,
                    # #       ndata_schemes={'metric': Scheme(shape=(128,), dtype=torch.float32), 'trace': Scheme(shape=(128,), dtype=torch.float32), 'log': Scheme(shape=(128,), dtype=torch.float32), 'root': Scheme(shape=(), dtype=torch.int64)}
                    # #       edata_schemes={})
                    # # Graph(num_nodes=8, num_edges=24,
                    # #       ndata_schemes={'metric': Scheme(shape=(128,), dtype=torch.float32), 'trace': Scheme(shape=(128,), dtype=torch.float32), 'log': Scheme(shape=(128,), dtype=torch.float32), 'root': Scheme(shape=(), dtype=torch.int64)}
                    # #       edata_schemes={})
                    # # 得到的每一个增强图，除了num_edges可能不同外，其他的都相同，且节点数都为8
                    # print(aug_graph.ndata["metric"].shape)  # torch.Size([8, 128])
                    # print(aug_graph.ndata["trace"].shape)   # torch.Size([8, 128])
                    # print(aug_graph.ndata["log"].shape) # torch.Size([8, 128])
                    # print("@" * 20)
                    aug_data.append((aug_graph, labels))    # 将增强后的图和标签添加到增强数据集中。

            # for time in range(6):
            #     for (graph, labels) in test_data:  # 通过丢弃节点进行数据增强。
            #         root = graph.ndata['root'].tolist().index(1)    # 找到根节点。
            #         aug_graph = aug_drop_node(graph, root, drop_percent=self.config.aug_percent)    # aug_graph：增强后的图   self.aug_percent = 0.2  # 数据增强的百分比，默认为 0.2。
            #         aug_data.append((aug_graph, labels))    # 将增强后的图和标签添加到增强数据集中。
        for (graph, labels) in train_data:  # 通过丢弃节点进行数据增强。
            aug_data.append((graph, labels))  # 将增强后的图和标签添加到增强数据集中。

        # i = 0
        # for (graph, labels) in test_data:  # 通过丢弃节点进行数据增强。
        #     if i < 100:
        #         i = i+1
        #         aug_data.append((graph, labels))  # 将增强后的图和标签添加到增强数据集中。

        # train_data and randomly sampled aug_data  创建训练数据的数据加载器，批量大小由配置参数指定。  # self.batch_size = 512
        train_dl = DataLoader(aug_data, batch_size=32, shuffle=True, collate_fn=self.collate)
        save_dataset(train_dl, 'aug_data')
        # val_dl = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=self.collate)
        # save_dataset(val_dl, 'val_data')
        test_dl = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=self.collate)
        save_dataset(test_dl, 'test_data')

        print("到此实现了数据集的增强，这里将数据集train_data的160个训练数据分别构建了10条增强的数据")

    def collate(self, samples): # samples：批次数据样本。    # batch是多个DGL图的列表
        graphs, labels = map(list, zip(*samples))   # 将图数据和标签分别提取出来。
        batched_graph = dgl.batch(graphs)   # 使用 dgl.batch 将多个图合并为一个批次图。
        batched_labels = torch.tensor(labels)   # 将标签转换为张量。
        return batched_graph, batched_labels

    def get_root_labels(self, nodes):
        labels2idx = {node: idx for idx, node in enumerate(nodes)}
        # 功能：创建一个字典，将每个节点名称映射到一个唯一的索引。enumerate(nodes)：遍历节点列表，同时获取索引和节点名称。 {node: idx}：将节点名称作为键，索引作为值，存储到字典中。
        # 如果 nodes = ['node1', 'node2', 'node3']，则 labels2idx 将是 { 'node1': 0, 'node2': 1, 'node3': 2 }。
        idx2label = {idx: label for idx, label in enumerate(nodes)}
        # 功能：创建一个字典，将每个索引映射回对应的节点名称。
        # 如果 nodes = ['node1', 'node2', 'node3']，则 idx2label 将是 { 0: 'node1', 1: 'node2', 2: 'node3' }。
        labels = np.array(self.labels['instance'].apply(lambda label_str: labels2idx[label_str]))
        # 将标签数据转换为索引形式  self.labels['instance'] 是 ['node1', 'node2', 'node1', 'node3']，则 labels 将是 [0, 1, 0, 2]。
        return labels, labels2idx, idx2label

    def get_type_labels(self, types):   # types：一个包含所有异常类型（anomaly_type）的列表。
        meta_labels = sorted(list(set(types)))  # 从输入的异常类型列表中提取唯一的异常类型，并对其进行排序。如果 types = ['type1', 'type2', 'type1', 'type3']，则 meta_labels 将是 ['type1', 'type2', 'type3']。
        labels2idx = {label: idx for idx, label in enumerate(meta_labels)}
        idx2label = {idx: label for idx, label in enumerate(meta_labels)}
        labels = np.array(self.labels['anomaly_type'].apply(lambda label_str: labels2idx[label_str]))
        return labels, labels2idx, idx2label

