import os
import time

import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from amfbo import PROJECT_ROOT
from amfbo.config.experiment_config import Config
from amfbo.core.losses.auto_weighted_loss import AutomaticWeightedLoss
from amfbo.core.losses.margin_loss import GaussianMarginLoss
from amfbo.core.models.amfbo_model import AMFBOModel
from amfbo.utils.early_stopping import EarlyStopping
from amfbo.utils.evaluation import *
from amfbo.utils.result_tracker import Result


class AMFBO(object):

    def __init__(self, config: Config, logger, log_dir: str):
        self.config = config
        self.logger = logger
        os.makedirs(log_dir, exist_ok=True)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("Currently using GPU {}".format(config.gpu_device))
            os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device
            self.device='cuda'
        else:
            logger.info("Currently using CPU (GPU is highly recommended)")
            self.device = 'cpu'
        self.result = Result()

        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def printParams(self):
        self.config.print_configs(self.logger)

    def train(self, aug_data):
        model = AMFBOModel(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.logger.info(model)
        self.logger.info(f"Start training for {self.config.epochs} epochs.")
        train_times = []

        earlyStop = EarlyStopping(patience=10, model=model)
        for epoch in range(self.config.epochs):
            print(epoch)
            epoch_loss = 0.0
            statr_time = time.time()
            train_num = 0.0

            for batch_graphs, batch_labels in aug_data:
                type_labels = batch_labels[:, 1]
                batch_graphs = batch_graphs.to(self.device)
                type_labels = type_labels.to(self.device)
                model.train()
                _, pre_sult, max_counts, importance_sums, l_rcl = model(batch_graphs)
                l_fti = F.cross_entropy(pre_sult, type_labels)
                total_loss = l_rcl + l_fti
                self.logger.debug("RCA_loss: {:.3f}, TC_loss: {:.3f}".format(l_rcl, l_fti))
                opt.zero_grad()
                total_loss.backward()
                for i in range(len(max_counts)):
                    feature = max_counts[i]
                    value = importance_sums[i]
                    k_value = compute_modulation_coedd(value)
                    if k_value != 0:
                        for name, param in model.named_parameters():
                            if param.grad is None:
                                continue
                            if feature in name:
                                param.grad = param.grad * k_value
                opt.step()
                epoch_loss += total_loss.detach().item() * pre_sult.size(0)
                train_num += pre_sult.size(0)
            mean_epoch_loss = epoch_loss / train_num
            end_time = time.time()
            train_times.append(end_time - statr_time)
            self.logger.info("Epoch {} done. Loss: {:.3f}".format(epoch, mean_epoch_loss))
            stop = earlyStop.should_stop(mean_epoch_loss, epoch, model)
            if stop:
                print(f"Early stop at epoch {earlyStop.bestepoch} due to lack of improvement.")
                self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                torch.save(earlyStop.best_model_wts, str(PROJECT_ROOT / 'result' / 'earlystop.pth'))
                break
        self.logger.info("Training has finished.")
        self.logger.debug(f"The training time is {np.sum(train_times)}[s]")
        self.logger.debug(f"The training time per epoch is {np.mean(train_times)}[s]")

    def evaluate(self, test_data, model=None):
        if model is None:
            model = AMFBOModel(self.config).to(self.device)
            model.load_state_dict(torch.load(str(PROJECT_ROOT / 'result' / 'earlystop.pth')))

        model.eval()
        inference_times = []
        rcl_results = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
        fti_results = {'pre': [], 'rec': [], 'f1': []}

        with torch.no_grad():
            for batch_graphs, batch_labels in test_data:
                type_labels = batch_labels[:, 1]
                batch_graphs = batch_graphs.to(self.device)
                type_labels = type_labels.to(self.device)

                model.eval()
                statr_time = time.time()
                root_logit, pre_sult, _, _, _ = model(batch_graphs)
                end_time = time.time()
                inference_times.append(end_time - statr_time)
                rcl_res = RCA_eval(root_logit, batch_graphs.batch_num_nodes(), batch_graphs.ndata['root'])
                fti_res = FTI_eval(pre_sult, type_labels)
                [rcl_results[key].append(value) for key, value in rcl_res.items()]
                [fti_results[key].append(value) for key, value in fti_res.items()]

        for k, v in rcl_results.items():
            rcl_results[k] = np.mean(v)
        for k, v in fti_results.items():
            fti_results[k] = np.mean(v)

        self.logger.debug(f"The test time is {np.sum(inference_times)}[s]")
        self.logger.debug(f"The test time per epoch is {np.mean(inference_times)}[s]")
        self.logger.info(
            "[Root localization] HR@1: {:.3%}, HR@2: {:.3%}, HR@3: {:.3%}, HR@4: {:.3%}, HR@5: {:.3%},  MRR@3: {:.3f}" \
            .format(rcl_results['HR@1'], rcl_results['HR@2'], rcl_results['HR@3'], rcl_results['HR@4'], rcl_results['HR@5'], rcl_results['MRR@3']))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}" \
                         .format(fti_results['pre'], fti_results['rec'], fti_results['f1']))

        print('HR@1：', rcl_results['HR@1'])
        print('HR@3：', rcl_results['HR@3'])
        print('avg@3', (rcl_results['HR@1']+rcl_results['HR@2']+rcl_results['HR@3']) / 3)
        print('MRR@3:', rcl_results['MRR@3'])
        print('precision:', fti_results['pre'])
        print('recall:', fti_results['rec'])
        print('f1-score:', fti_results['f1'])
    
    def cal_rcl_loss(self, root_logit, batch_graphs):
        num_nodes_list = batch_graphs.batch_num_nodes()
        total_loss = None
        
        start_idx = 0
        for idx, num_nodes in enumerate(num_nodes_list):
            end_idx = start_idx + num_nodes
            node_logits = root_logit[start_idx : end_idx].reshape(1, -1)
            root = batch_graphs.ndata["root"][start_idx : end_idx].tolist().index(1)
            loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(self.device))
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            start_idx += num_nodes
        l_rcl = total_loss / len(num_nodes_list)
        return l_rcl
