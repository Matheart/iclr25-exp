import os
import sys
import logging
import torch
import torch.nn.functional as F
import random
import numpy as np
import wandb
import queue
import datetime
import math
from pyhessian import hessian
from sklearn.metrics import mutual_info_score
from config import get_cfg_defaults
import utils
import datasets
from models import get_model
import secrets
import ipdb
import pickle
import copy
import math

logger = logging.getLogger(__name__)

def get_labels(y_pred, cfg):
    if len(y_pred.size()) == 1:
        threshold = cfg.OPT.ClSThrsh
        labels = torch.zeros_like(y_pred)
        labels[y_pred>=threshold] = cfg.DATA.BINARY_LABELS[1]*torch.ones_like(y_pred[y_pred>=threshold]) # the positive label
        labels[y_pred<threshold] = cfg.DATA.BINARY_LABELS[0]*torch.ones_like(y_pred[y_pred<threshold]) # the negative label
        return labels

    # if output layer > 1 then argmax over output dist to get class labels
    # assuming shape of [N, OUT_DIM]
    return torch.argmax(y_pred, 1)

def compute_accuracy(y_pred, y_true, cfg):
    labels = get_labels(y_pred, cfg)
    if cfg.DATA.ONE_HOT_ENCODE:
        y_true = torch.argmax(y_true, 1)
    correct = torch.sum(labels == y_true)
    return correct / y_true.size(0)

def update_lr(optimizer, delta_lr):
    '''
    updates the LR as a linear function of itself and delta_lr
    new_lr = old_lr + delta_lr
    '''
    for g in optimizer.param_groups:
        g['lr'] += delta_lr

def run_epoch(network, loss_fn, optimizer, train_loader, test_loaders, \
              epoch_idx, curr_step, total_steps, delta_lr, total_warmup_steps, \
              device, cfg, run, train_loss_queue):
    network.train()
    for batch_idx, (batchX, batchY) in enumerate(train_loader):
        batchX, batchY = batchX.to(device), batchY.to(device)

        if cfg.DATA.FLATTEN_INPUT and cfg.DATA.DATASET != 'cifar5m-flat':
            batchX = torch.flatten(batchX, start_dim=cfg.DATA.FLATTEN_START_DIM)

        y_pred = network(batchX)

        if cfg.OPT.LOSS_FN == "CrossEntropy":
            batchY = batchY.long()
        else:
            if cfg.DATA.ONE_HOT_ENCODE:
                batchY = F.one_hot(batchY, num_classes=cfg.MODEL.OUT_DIM)
            batchY = batchY.float()
            y_pred = y_pred.squeeze()

        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        if cfg.OPT.GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.OPT.GRAD_CLIP)
        optimizer.step()
        if curr_step < total_warmup_steps:
            update_lr(optimizer, delta_lr)
        curr_train_loss = loss.detach()
        curr_train_acc = compute_accuracy(y_pred, batchY, cfg).detach()

        if cfg.SYSTEM.TEST_FREQ >= 0 and curr_step % cfg.SYSTEM.TEST_FREQ == 0:
            test(network, loss_fn, test_loaders, epoch_idx, curr_step, total_steps, device, cfg, run)

        if curr_step % cfg.LOGGER.LOG_FREQ == 0:
            run.log({'train_loss_batch': curr_train_loss, 'train_accu_batch': curr_train_acc}, step=curr_step)
            logger.info('Epoch %d, batch %d/%d, step %d/%d, train_loss_batch = %f, train_accu_batch = %f' % (
                epoch_idx, batch_idx, len(train_loader), curr_step, total_steps, curr_train_loss, curr_train_acc,
            ))

        # pop the oldest loss value from the front of the queue and add new loss
        if train_loss_queue.full():
            train_loss_queue.get()
        train_loss_queue.put(curr_train_loss.cpu())

        if np.average(train_loss_queue.queue) <= cfg.OPT.TRAIN_LOSS_EPSILON_STOP:
            return curr_step, train_loss_queue

        curr_step += 1
    return curr_step, train_loss_queue

def test(network, loss_fn, test_loaders, epoch_idx, curr_step, total_steps, device, cfg, run, name = "test"):
    network.eval()
    results = {}
    for test_prefix, test_loader in test_loaders:
        test_loss = 0.0
        test_correct = 0
        f_hat = []
        n_data = 0
        test_excess_risk = 0.0 # tracking f(x)^2 over test set, only makes sense if 1-dim output and true regression fn is 0 fn
        test_mutual_info = 0.0
        all_pred_labels = None
        all_true_labels = None

        if cfg.SYSTEM.COMPUTE_HESSIAN:
            hessian_loader = [(torch.flatten(X, start_dim=cfg.DATA.FLATTEN_START_DIM).float(), F.one_hot(Y, num_classes=cfg.MODEL.OUT_DIM).float() if cfg.DATA.ONE_HOT_ENCODE else Y) for (X, Y) in test_loader]
            hessian_comp = hessian(network, loss_fn, dataloader=hessian_loader, cuda=cfg.SYSTEM.DEVICE == 'cuda')
            top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=10)
            top_eigenvalues = sorted(top_eigenvalues, reverse=True)
            run.log({
                   test_prefix + name+'_hessian_spectrum': np.array(top_eigenvalues)
               }, step=curr_step)


        for (batchX, batchY) in test_loader:
            batchX, batchY = batchX.to(device), batchY.to(device)

            if cfg.DATA.FLATTEN_INPUT:
                batchX = torch.flatten(batchX, start_dim=cfg.DATA.FLATTEN_START_DIM)
            y_pred = network(batchX)
            f_hat = y_pred.view(-1,).tolist() + f_hat

            if cfg.OPT.LOSS_FN == "CrossEntropy":
                batchY = batchY.long()
            else:
                if cfg.DATA.ONE_HOT_ENCODE:
                    batchY = F.one_hot(batchY, num_classes=cfg.MODEL.OUT_DIM)
                batchY = batchY.float()
                y_pred = y_pred.squeeze()

            loss = loss_fn(y_pred, batchY)

            batch_size = batchY.size(0)
            test_correct += compute_accuracy(y_pred, batchY, cfg).detach()*batch_size
            n_data += batch_size
            test_loss += loss.detach() * batch_size
            bayes_optimal = 0.0
            #bayes_optimal = 1-2*cfg.DATA.LABEL_NOISE
            #bayes_optimal = cfg.DATA.LABEL_PROB*1+(1-cfg.DATA.LABEL_PROB)*(-1)
            test_excess_risk += torch.mean(torch.square(y_pred-bayes_optimal)).detach()*batch_size

            '''
            mutual info computations, ignore for now
            if cfg.DATA.ONE_HOT_ENCODE:
                batchY = torch.argmax(batchY, 1)
            if all_pred_labels is None:
                all_pred_labels = get_labels(y_pred, cfg)
                all_true_labels = copy.deepcopy(batchY)
            else:
                pred_labels = get_labels(y_pred, cfg)
                all_pred_labels = torch.cat((all_pred_labels, pred_labels), dim=0)
                all_true_labels = torch.cat((all_true_labels, batchY), dim=0)
            '''

        #test_mutual_info += mutual_info_score(all_true_labels.cpu(), all_pred_labels.cpu())

        results[f'{test_prefix}test_acc'] = test_correct / n_data
        results[f'{test_prefix}test_loss'] = test_loss / n_data
        results[f'{test_prefix}test_excess_risk']  = test_excess_risk/ n_data
        results[f'{test_prefix}test_variance'] = np.var(np.array(f_hat))
        results[f'{test_prefix}test_expectation'] = np.mean(np.array(f_hat))
        #results[f'{test_prefix}test_mutual_info'] = test_mutual_info / n_data
        #results['f_hat'] = np.array(f_hat)
        results[f'{test_prefix}lab_prob'] = cfg.DATA.LABEL_PROB

        run.log({
            test_prefix + name+'_loss': results[f'{test_prefix}test_loss'],
            test_prefix + name+'_excess_risk': results[f'{test_prefix}test_excess_risk'] ,
            test_prefix + name+'_accu': results[f'{test_prefix}test_acc'],
            test_prefix + name+'_variance': results[f'{test_prefix}test_variance'],
            test_prefix + name+'_expectation':results[f'{test_prefix}test_expectation'],
            #name+'_f_hat':results['f_hat'],
            test_prefix + 'lab_prob' : results[f'{test_prefix}lab_prob']
            #test_prefix + name+'_mutual_info': results[f'{test_prefix}test_mutual_info']
        }, step=curr_step)

        #if name == 'test':
        #    run.log({
        #        test_prefix + name+'_histo': np.array(f_hat)
        #    }, step=curr_step)

        logger.info('Epoch %d, step %d/%d, %s loss = %f, %s excess_risk = %f, %s_Accu = %f' % (
            epoch_idx, curr_step, total_steps,test_prefix+name, results[f'{test_prefix}test_loss'],test_prefix+name, results[f'{test_prefix}test_excess_risk'],test_prefix+name, results[f'{test_prefix}test_acc'],
        ))


    return results

def setup(cfg):
    logging.basicConfig(format=cfg.LOGGER.FORMAT, \
                        level=cfg.LOGGER.LEVEL)
    if cfg.SYSTEM.SEED:
        utils.set_seeds(cfg.SYSTEM.SEED)
    utils.print_cfg(cfg)

    os.environ["WANDB_API_KEY"] = secrets.WANDB_API_KEY
    os.environ["WANDB_MODE"] = cfg.WANDB.MODE
    timestr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wandb_name = '%s-%s-train_n: %i; label_noise: %.2f; noise_variance: %.2f; inp_dim: %i' % (cfg.MODEL.MODEL_TYPE, cfg.DATA.DATASET,cfg.DATA.TRAIN_N, cfg.DATA.LABEL_NOISE, cfg.DATA.NOISE_VARIANCE, cfg.MODEL.INP_DIM)

    run = wandb.init(project=cfg.WANDB.PROJECT_NAME, \
                     entity=cfg.WANDB.ORG)
    # we'll use the run_id as the name and correlate it with filesystem organization
    run.name = run.id + " - "+wandb_name
    run.save()
    # for now we are overriding wandb config with our config
    # but this can also go the other way around if it's easier
    utils.update_wandb_config(cfg, run.config)

    return run

def main(config_file=None, argv=None,manual_config_override = None):
    cfg = get_cfg_defaults()
    if manual_config_override != None:
        cfg = manual_config_override
    if config_file is not None:
        print(f'Using config_file: {config_file}')
        cfg.merge_from_file(config_file)
    # optionally allow config file path to be passed in as first CLI argument:
    # python3 train.py --config_file /path/to/file.yaml
    if argv != None and len(argv) > 0:
        if argv[1] == '--config_file':
            print(f'Using config_file: {argv[2]}')
            cfg.merge_from_file(argv[2])
            # merge the remaining CLI overrides after merging from file
            if len(argv) > 3:
                cfg.merge_from_list(argv[3:])
        else:
            argv = argv[1:]
            cfg.merge_from_list(argv)
    cfg.freeze()

    run = setup(cfg)
    # add some summary variables for plotting
    run.summary['train_n'] = cfg.DATA.TRAIN_N

    logger.info('loading network')
    network = get_model(cfg)
    logger.info(network)

    logger.info(f'Setting device: {cfg.SYSTEM.DEVICE}')
    device = torch.device(cfg.SYSTEM.DEVICE)
    network.to(device)

    logger.info('Preparing train/test datasets and dataloaders')
    # when using flattened cifar5m if model has lower inp dim than 3072 then
    # we will do dim reduction on each batch using PCA vectors
    pca_vectors = None
    if cfg.DATA.DATASET == 'cifar5m-flat':
        with open(cfg.DATA.CF5M_PCA_VECTORS, 'rb') as fp:
            pca_vectors = pickle.load(fp)

    train_loader, test_loaders = datasets.load(cfg.DATA, cfg.MODEL, utils.seed_worker, pca_vectors=pca_vectors)

    if cfg.OPT.LOSS_FN == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction=cfg.OPT.LOSS_FN_REDUCTION)
    elif cfg.OPT.LOSS_FN == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=cfg.OPT.LOSS_FN_REDUCTION)
    else:
        raise Exception('Unsupported loss fn: %s' % (cfg.OPT.LOSS_FN))

    wd_params, no_wd_params = utils.get_wd_params(network)

    if cfg.OPT.OPT_ALG == 'sgd':
        optimizer = torch.optim.SGD([{'params': no_wd_params, 'weight_decay': 0.0},
                                     {'params': wd_params}],
                                    lr=cfg.OPT.LEARNING_RATE, momentum=cfg.OPT.MOMENTUM,
                                    dampening=cfg.OPT.DAMPENING, weight_decay=cfg.OPT.WEIGHT_DECAY,
                                    nesterov=cfg.OPT.NESTEROV)
    elif cfg.OPT.OPT_ALG == 'adam':
        optimizer = torch.optim.Adam([{'params': no_wd_params, 'weight_decay': 0.0},
                                      {'params': wd_params}],
                                     lr=cfg.OPT.LEARNING_RATE, weight_decay=cfg.OPT.WEIGHT_DECAY)
    elif cfg.OPT.OPT_ALG == 'adamw':
        optimizer = torch.optim.AdamW([{'params': no_wd_params, 'weight_decay': 0.0},
                                       {'params': wd_params}],
                                     lr=cfg.OPT.LEARNING_RATE, weight_decay=cfg.OPT.WEIGHT_DECAY)
    else:
        raise Exception('Unsupported optimization alg: %s' % (cfg.OPT.OPT_ALG))

    # wandb.watch(network, loss_fn, 'all') # can comment or uncomment to log or not log gradients and parameters from training to Weights & Biases
    curr_step = 0 # counting number of SGD steps taken over full training
    total_steps = cfg.OPT.TOTAL_EPOCHS * len(train_loader) # total number of SGD steps to take
    total_warmup_steps = 0
    delta_lr = 0.0
    if cfg.OPT.WARMUP_LR:
        total_warmup_steps = cfg.OPT.WARMUP_EPOCHS * len(train_loader)
        delta_lr = (cfg.OPT.LEARNING_RATE - 1e-8) / total_warmup_steps
        update_lr(optimizer, 1e-8-cfg.OPT.LEARNING_RATE)

    scheduler = None
    if cfg.OPT.LR_SCHEDULE == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.OPT.TOTAL_EPOCHS-cfg.OPT.WARMUP_EPOCHS)
    elif cfg.OPT.LR_SCHEDULE == 'multi_step_lr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.OPT.LR_GAMMA, milestones=cfg.OPT.LR_MILESTONES)

    # do a test at initialization
    test(network, loss_fn, test_loaders, -1, -1, total_steps, device, cfg, run, name = "test")

    # track training loss over last ten batches (FIFO queue) for when TOTAL_EPOCHS == -1
    # then we train until train loss hits epsilon
    train_loss_queue = queue.Queue(maxsize=10)
    epoch_idx = 0
    while (epoch_idx < cfg.OPT.TOTAL_EPOCHS) or \
          (cfg.OPT.TOTAL_EPOCHS == -1 and np.average(train_loss_queue.queue) > cfg.OPT.TRAIN_LOSS_EPSILON_STOP) or \
          (epoch_idx == 0):
        run.log({
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=curr_step)
        curr_step, train_loss_queue = run_epoch(network, loss_fn, optimizer, train_loader, \
                                                test_loaders, epoch_idx, curr_step, total_steps, \
                                                delta_lr, total_warmup_steps, device, cfg, run, \
                                                train_loss_queue)

        if scheduler and curr_step > total_warmup_steps:
            scheduler.step()

        ### log performance on the whole training data set after each epoch
        test(network, loss_fn, [('', train_loader)], epoch_idx, curr_step, total_steps, device, cfg, run, name = "train")
        if epoch_idx == 0 or cfg.SYSTEM.TEST_FREQ <= 0:
            results = test(network, loss_fn, test_loaders, epoch_idx, curr_step, total_steps, device, cfg, run, name = "test")
            utils.log_prefixed_summary(run, f'epoch{epoch_idx+1}', results)
        epoch_idx += 1

    epoch_idx -= 1 # will end up overcounting by 1 in the while loop
    model_filepath = os.path.join(cfg.MODEL.BASE_OUT_DIR, cfg.WANDB.PROJECT_NAME, str(run.id))
    os.makedirs(model_filepath, exist_ok=True)
    utils.save_model(network, model_filepath, name='model.pt')

    results  = \
        test(network, loss_fn, test_loaders, epoch_idx, curr_step, total_steps, device, cfg, run, name = "test")

    utils.log_prefixed_summary(run, 'final', results)
    run.finish()

if __name__=='__main__':
    #config_file = '../../configs/mlp_binary_mnist.yaml'
    config_file = None
    main(argv=sys.argv, config_file=config_file)
