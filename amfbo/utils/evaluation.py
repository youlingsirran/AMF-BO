import math

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def RCA_eval(root_logit, num_nodes_list, roots):
    res = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
    
    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
        root = roots[start_idx : end_idx].tolist().index(1)

        _, sorted_indices = torch.sort(node_logits, descending=True)
        for j in range(1, 6):
            # HR@k
            if root in sorted_indices.flatten()[:j]:
                res[f"HR@{j}"].append(1)
            else:
                res[f"HR@{j}"].append(0)
        rank = (sorted_indices == root).nonzero(as_tuple=True)[1].item() + 1
        if rank <= 3:
            res["MRR@3"].append(1 / rank)
        else:
            res["MRR@3"].append(0)

        start_idx += num_nodes
    for k in range(1, 6):
        res[f'HR@{k}'] = np.sum(res[f'HR@{k}'])/len(num_nodes_list)
    res['MRR@3'] = np.sum(res['MRR@3'])/len(num_nodes_list)
    return res


def RCA_single_modality(root_logit, num_nodes_list, roots):
    res = {"HR@1": []}
    rcl_results = {"HR@1": []}

    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx: end_idx].reshape(1, -1)
        root = roots[start_idx: end_idx].tolist().index(1)
        _, sorted_indices = torch.sort(node_logits, descending=True)
        for j in range(1, 2):
            # HR@k
            if root in sorted_indices.flatten()[:j]:
                res[f"HR@{j}"].append(1)
            else:
                res[f"HR@{j}"].append(0)

        start_idx += num_nodes
    for k in range(1, 2):
        res[f'HR@{k}'] = np.sum(res[f'HR@{k}']) / len(num_nodes_list)

    [rcl_results[key].append(value) for key, value in res.items()]
    for k, v in rcl_results.items():
        rcl_results[k] = np.mean(v)

    return rcl_results['HR@1']

def FTI_eval(output, target, k=5):
    res = {"pre": [], "rec": [], "f1": []}
    res['pre']=precision(output, target, k)
    res['rec']=recall(output, target, k)
    res['f1']=2 * res['pre'] * res['rec'] / (res['pre'] + res['rec'])
    return res


def target_rank(output, target, k=10):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    ranks = []
    for col in range(correct.size(1)):
        try:
            idx=torch.where(correct[:, col] == target[col])[0].item() + 1
        except:
            idx=10
        ranks.append(idx)
    
    return ranks

def precision(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average='weighted')

    return pre

def recall(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    rec = recall_score(y_true, y_pred[:, 0], average='weighted')

    return rec

def f1score(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_true, y_pred[:, 0], average='weighted')

    return f1

def compute_modulation_coedd(a):

    # max_val = max(a, b, c)
    # if a != max_val or a == b or a == c:
    #     return 0
    #
    # total = a + b + c
    # normalized_a = a / total
    # # normalized_b = b / total
    # # normalized_c = c / total

    tanh_input = 0.9 * (a * 3 - 1)
    # tanh_input = 0.5 * a
    # tanh_input = 0.2 * (normalized_a * 3 - 1)
    modulation_value = 1 - math.tanh(tanh_input)

    return modulation_value
