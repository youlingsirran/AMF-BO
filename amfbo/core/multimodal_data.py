from torch.utils.data import Dataset

from amfbo.core.augmentation import *
from amfbo.core.models.backbone.dominant import (cohesion_early_fusion,
                                                 cohesion_early_fusion_2)


class MultiModalDataset(Dataset):
    def __init__(self):
        self.data = []

    def add_data(self, metric_Xs, trace_Xs, log_Xs, metric_Xs_word, trace_Xs_word, log_Xs_word, global_root_id, failure_type_id, local_root, nodes, edges):
        node_num = len(nodes)
        graph = dgl.graph(edges, num_nodes=node_num)
        graph.ndata["metric"] = torch.FloatTensor(metric_Xs)
        graph.ndata["trace"] = torch.FloatTensor(trace_Xs)
        # print(graph.ndata["trace"].shape)   # torch.Size([10, 128])
        graph.ndata["log"] = torch.FloatTensor(log_Xs)
        # print(graph.ndata["log"].shape)     # torch.Size([10, 128])
        # graph.ndata["logs"] = torch.zeros(logs[i].shape)
        metric_Xs_word_cpu = []
        for tensor in metric_Xs_word:
            if isinstance(tensor, torch.Tensor):
                metric_Xs_word_cpu.append(tensor.cpu())
            else:
                metric_Xs_word_cpu.append(torch.from_numpy(tensor))
        metric_word = torch.stack(metric_Xs_word_cpu)

        trace_Xs_word_cpu = []
        for tensor in trace_Xs_word:
            if isinstance(tensor, torch.Tensor):
                trace_Xs_word_cpu.append(tensor.cpu())
            else:
                trace_Xs_word_cpu.append(torch.from_numpy(tensor))
        trace_word = torch.stack(trace_Xs_word_cpu)
        log_Xs_word_cpu = []
        for tensor in log_Xs_word:
            if isinstance(tensor, torch.Tensor):
                log_Xs_word_cpu.append(tensor.cpu())
            else:
                log_Xs_word_cpu.append(torch.from_numpy(tensor))
        log_word = torch.stack(log_Xs_word_cpu)

        # metric_word, trace_word, log_word= cohesion_early_fusion_2(metric_word, trace_word, log_word)
        graph.ndata["metric_word"] = metric_word
        graph.ndata["trace_word"] = trace_word
        graph.ndata["log_word"] = log_word

        root_labels = [0] * len(nodes)
        root_labels[nodes.index(local_root)] = 1

        graph.ndata["root"] = torch.LongTensor(root_labels)

        in_degrees = graph.in_degrees()
        zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
        for node in zero_indegree_nodes:
            graph.add_edges(node, node)

        # Graph(num_nodes=10, num_edges=39,
        #       ndata_schemes={'metric': Scheme(shape=(128,), dtype=torch.float32), 'trace': Scheme(shape=(128,), dtype=torch.float32), 'log': Scheme(shape=(128,), dtype=torch.float32), 'root': Scheme(shape=(), dtype=torch.int64)}
        #       edata_schemes={})

        self.data.append((graph, (global_root_id, failure_type_id)))
           

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
