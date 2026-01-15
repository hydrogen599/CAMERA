from ast import arguments
import os
import sys
import time
import random
import argparse
import torch
import numpy as np
# For DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import Config,Logger
from Ampmm_base.runner import Runner

local_rank = 0
seed = 42
def parse_args():
    parser = argparse.ArgumentParser(description='Training And Test')

    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--mode', default='train', help='train or test')

    parser.add_argument('--pair', type=float)
    parser.add_argument('--mse', type=float)

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--task', type=str, default='rank')
    args = parser.parse_args()
    return args

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def get_config_and_logger(config, 
          pair=None, mse=None,
          epochs=None, batch_size=None, 
          mode='train', dataset=None, model=None, task=None):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    cfg = Config.fromfile(config)
    if dataset is not None:
        cfg.dataset_name = dataset
    if model is not None:
        cfg.model['name'] = model
    if task is not None:
        cfg.task = task

    if batch_size is not None:
        cfg.batch_size = batch_size
        cfg.data['train']['batch_per_gpu'] = batch_size
    else:
        cfg.batch_size = cfg.data['train']['batch_per_gpu']
        batch_size = cfg.data['train']['batch_per_gpu']
    if epochs is not None:
        cfg.epochs = epochs
    else:
        epochs = cfg.epochs

    if pair is not None:
        cfg.model['losses']['Ensemble_ContrastiveLoss']['weight'] = pair
    else:
        pair = cfg.model['losses']['Ensemble_ContrastiveLoss']['weight']
    if mse is not None:
        cfg.model['losses']['Ensemble_MSELoss']['weight'] = mse
    else:
        mse = cfg.model['losses']['Ensemble_MSELoss']['weight']

    cfg.mse = mse
    cfg.pair = pair

    cfg.local_rank = local_rank
    cfg.ckpt_path = os.path.join(cfg.work_dir,cfg.dataset_name, mode, 'final.ckpt')    
    
    # create logger and work_dir
    if dist.get_rank() == 0:
        cfg.work_dir = os.path.join(cfg.work_dir,cfg.dataset_name,mode)
        os.makedirs(cfg.work_dir,exist_ok=True)
    logger = Logger(cfg.work_dir)

    return cfg, logger

if __name__ == '__main__':
    args = parse_args()
    config = args.config

    local_rank = args.local_rank
    seed = args.seed
    if seed is not None:
        set_seed(seed)

    cfg, logger = get_config_and_logger(config, 
                        pair=args.pair, mse=args.mse,
                        epochs=args.epochs, batch_size=args.batch_size, 
                        mode=args.mode,
                        dataset=args.dataset, model=args.model, task=args.task)
    
    if local_rank == 0:
        logger.info("Running with :\nmodel_name:{}, task:{}".format(cfg.model['name'], cfg.task))
        
    runner = Runner(cfg,logger,args.local_rank,args.mode)
    if args.mode == 'train':
        runner.run()
    elif args.mode == 'test':
        runner.test()
    else:
        if args.local_rank == 0:
            print("Please ensure args.mode to be train or test")
            exit()