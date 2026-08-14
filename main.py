import random
import warnings
from pathlib import Path

import numpy as np
import torch

from amfbo import PROJECT_ROOT
from amfbo.config.experiment_config import Config
from amfbo.core.trainer import AMFBO
from amfbo.pipeline.event_pipeline import EventPipeline
from amfbo.utils.dataset_io import load_dataset
from amfbo.utils.logger import get_logger

warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def build_dataloader(config: Config, logger, exp_name):
    reconstruct = config.reconstruct
    processor = EventPipeline(config, logger)
    processor.process(reconstruct=reconstruct, exp_name=exp_name)

def train_and_evaluate(config: Config, log_dir, exp_name):
    set_seed(2)
    logger = get_logger(log_dir, exp_name)
    logger.info("Load dataset")
    # build_dataloader(config, logger, exp_name)
    aug_data = load_dataset("aug_data", exp_name)
    model = AMFBO(config, logger, log_dir)
    logger.info("Training...")
    model.train(aug_data)
    test_data = load_dataset("test_data", exp_name)
    model.evaluate(test_data)

if __name__ == '__main__':
    for dataset in ['gaia']:
        config = Config(dataset)
        config.reconstruct = True
        train_and_evaluate(config, str(PROJECT_ROOT / 'logs' / dataset), dataset)
