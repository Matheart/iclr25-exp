import logging
import torch
import torch.nn as nn
import random
import numpy as np
import os

logger = logging.getLogger(__name__)

def log_prefixed_summary(run, prefix, summary_dict):
    for (key, value) in summary_dict.items():
        run.summary[f'{prefix}_{key}'] = value

def save_model(model, out_path, name='model.pt'):
    def unwrap_model(model): # unwraps DataParallel, etc
        return model.module if hasattr(model, 'module') else model
    os.makedirs(out_path, exist_ok=True)
    file_path = os.path.join(out_path, name)
    torch.save(unwrap_model(model).state_dict(), file_path)

def load_state_dict(model, filepath):
    model.load_state_dict(torch.load(file_path))

def update_wandb_config(cfg, run_config, key_prefix=''):
    for key, value in cfg.items():
        if isinstance(value, dict):
            update_wandb_config(value, run_config, key_prefix=key_prefix+str(key)+'.')
        else:
            run_config.update({f'{key_prefix}{key}': f'{value}'})

def set_seeds(seed):
    logger.info('Setting seeds')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        logger.info('CUDA is available, seeding CUDA')
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

'''
Pass this to the worker_init_fn in the data loader
to have deterministic behavior in data loading
when utilizing multiple workers.
'''
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def print_cfg(cfg, key_prefix=''):
    '''
    Iterate the values of a dictionary and all nested dictionary values
    and print the key, value pairs
    '''
    for key, value in cfg.items():
        if isinstance(value, dict):
            print_cfg(value, key_prefix=key_prefix+str(key)+'.')
        else:
            logger.info(f'{key_prefix}{key} = {value}')

@torch.no_grad()
def get_wd_params(model):
    # Source: https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/8
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if hasattr(param,'requires_grad') and not param.requires_grad:
            continue
        if 'weight' in name and 'norm' not in name and 'bn' not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return decay, no_decay
