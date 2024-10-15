import logging
import torch
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()

## Logger Configurations
_C.LOGGER = CN()
_C.LOGGER.LEVEL = logging.INFO # supported levels: INFO, DEBUG, WARNING, ERROR
_C.LOGGER.FORMAT = '%(filename)s:%(lineno)d (%(name)s) | %(levelname)s [%(asctime)s]: %(message)s'
_C.LOGGER.LOG_FREQ = 50 # frequency to print train logs in terms of SGD steps

#DATA_N/BATCH_SIZE # frequency in SGD steps to run test set eval
# alternatively, set = 'epoch' to run test eval once per epoch
_C.SYSTEM.TEST_FREQ = -1
_C.SYSTEM.COMPUTE_HESSIAN = False
_C.SYSTEM.COMPUTE_MUTUAL_INFO = False

## Weights and Biases (wandb) Configurations
_C.WANDB = CN()
_C.WANDB.PROJECT_NAME = 'ANON'
_C.WANDB.ORG = 'ANON'
_C.WANDB.MODE = 'online' # can specify online, offline, dryrun, etc

## Data Configurations
_C.DATA = CN()
_C.DATA.ROOT = 'ANON'
_C.DATA.DATASET = 'cifar5m-flat' #'synthetic1'# 'CIFAR10'

_C.DATA.DATA_N = 1000
_C.DATA.TEST_N = 10000 # number of test samples
_C.DATA.IMAGE_SIZE = 32


_C.DATA.NORM_SCALE = CN()
# set values we know when using standard datasets
if _C.DATA.DATASET == 'MNIST':
    _C.DATA.DATA_N = 60000
    _C.DATA.TEST_N = 10000
    _C.DATA.IMAGE_SIZE = 28

    _C.DATA.NORM_SCALE.mean = (0.1307,)
    _C.DATA.NORM_SCALE.std = (0.3081,)
if _C.DATA.DATASET == 'CIFAR10':
    _C.DATA.DATA_N = 50000
    _C.DATA.TEST_N = 10000

    _C.DATA.NORM_SCALE.mean = (0.4914, 0.4822, 0.4465)
    _C.DATA.NORM_SCALE.std = (0.2023, 0.1994, 0.2010)
elif _C.DATA.DATASET == 'cifar5m' or _C.DATA.DATASET == 'cifar5m-flat':
    _C.DATA.DATA_N = 5002240
    _C.DATA.TEST_N = 10000
    
    _C.DATA.NORM_SCALE.mean = (0.4914, 0.4822, 0.4465)
    _C.DATA.NORM_SCALE.std = (0.2023, 0.1994, 0.2010)

_C.DATA.BATCH_SIZE = min(64,_C.DATA.DATA_N)

# optionally subsample the training data by subsample_ratio
_C.DATA.DATASET_SUBSAMPLE_RATIO = 1.0 # percent of training to subsample, set in such a way that this variable and TOTAL_EPOCHS are integer valued (1.0 for all of the data)
_C.DATA.TRAIN_N = int(_C.DATA.DATA_N * _C.DATA.DATASET_SUBSAMPLE_RATIO)

# if you have data like [batch_size, x_dim, y_dim, z_dim, etc] and want to flatten to [batch_size, x_dim*y_dim*z_dim*etc]
# this is useful if you want to flatten images (or non-standard data loading) for FF networks
# if using convolutional nets this should be False for image data
_C.DATA.FLATTEN_INPUT = False
_C.DATA.FLATTEN_START_DIM = 1 # what dimension to start flattening, usually 1 as dim=0 is batch dim
_C.DATA.NUM_WORKERS = 1
_C.DATA.ONE_HOT_ENCODE = False

# label randomization settings
_C.DATA.LABEL_FN = 'square' # options: ['uniform', 'square', 'constant']
_C.DATA.RANDOMIZE_LABELS = True # for now this will generate random binary label vectors
_C.DATA.LABEL_PROB = 0.7 # probability for label 1
_C.DATA.LABEL_SCALE = int(1) # if randomized labels, this is a multiplicative factor on labels
if _C.DATA.RANDOMIZE_LABELS == False:
    _C.DATA.LABEL_SCALE = int(1)
_C.DATA.BINARY_LABELS = [-_C.DATA.LABEL_SCALE, _C.DATA.LABEL_SCALE] # if using: always have negative class in slot 0 and positive in slot 1 (usually either [-1, 1] or [0, 1])
_C.DATA.LABEL_NOISE = 0.1 # label noise percentage in [0, 1], set to 0 for regression problems
_C.DATA.EVAL_NOISY_TEST = True # when using non-zero label noise in classification, whether or not to eval on the noisy test as well
_C.DATA.NOISE_VARIANCE = 1.0

_C.DATA.CF5M_PCA_VECTORS = 'ANON'

## Model Configurations
_C.MODEL = CN()
_C.MODEL.BASE_OUT_DIR = 'ANON'
_C.MODEL.MODEL_TYPE = 'feedforward'#'resnet18' # if this is 'resnet18' or another standard architecture, some custom settings here are ignored
# if using feed forward specify dim (inp and hid), if using custom CNN specify channels (i.e. inp is usually 3 for RGB)

_C.DATA.DIM_Syntc = [100] # need a default setting, override for specific sets
if _C.DATA.DATASET == 'synthetic1':
    if _C.MODEL.MODEL_TYPE == 'resnet18':
        _C.DATA.DIM_Syntc = [3,32,32]
    else:
        _C.DATA.DIM_Syntc = [100]
    _C.MODEL.INP_DIM = _C.DATA.DIM_Syntc[0]
elif _C.DATA.DATASET == 'MNIST':
    _C.MODEL.INP_DIM = 28*28
elif _C.DATA.DATASET == 'CIFAR':
    _C.MODEL.INP_DIM = 32*32*3
elif _C.DATA.DATASET == 'cifar5m-flat':
    # optionally specify _C.MODEL.INP_DIM <= 3072 to perform dim reduction
    # on cifar5m flat vectors
    _C.MODEL.INP_DIM = 32*32*3


# INP_DIM =  # 3*32*32 #28*28 # MNIST = 28*28, CIFAR = 3*32*32
_C.MODEL.HID_DIM = 512
_C.MODEL.OUT_DIM = 1 # if RANDOMIZE_LABELS is True or ONE_HOT_ENCODE is False with MSE loss this should be 1, if using ONE_HOT_ENCODE or CE loss this should = N_CLASSES
_C.MODEL.N_HID_LAYERS = 3 # should be >=1, i.e. if =1 then net is inp -> hid -> out, if =2 then net is inp -> hid -> hid -> out
_C.MODEL.FIX_BACKBONE = False # if True then only output linear layer is trained (in feedforward)
# dictionary of init type and any relevant parameters, for default init use:
_C.MODEL.ACT_FN = 'relu'
_C.MODEL.DROPOUT_RATE = 0.0

_C.MODEL.INIT = CN()
_C.MODEL.INIT.type = 'default'
# _C.MODEL.INIT.type = 'normal'
_C.MODEL.INIT.mu = 0.0
_C.MODEL.INIT.std = 0.1
_C.MODEL.INIT.bias = 1.0

## Optimization Configurations
_C.OPT = CN()
_C.OPT.LOSS_FN = 'MSE'
_C.OPT.ClSThrsh = 0
_C.OPT.LOSS_FN_REDUCTION = 'mean' # 'mean' or 'sum'
_C.OPT.OPT_ALG = 'sgd'
_C.OPT.LR_SCHEDULE = 'None'#'cosine', or 'multi_step_lr'
_C.OPT.LR_MILESTONES = [42] # used when LR_SCHEDULE is 'multi_step_lr'
_C.OPT.LEARNING_RATE = 1e-2
_C.OPT.LR_GAMMA = 0.1
_C.OPT.WARMUP_LR = True
_C.OPT.WARMUP_EPOCHS = 10
_C.OPT.MOMENTUM = 0.0
_C.OPT.DAMPENING = 0.0
_C.OPT.WEIGHT_DECAY = 0.0#1e-4
_C.OPT.NESTEROV = False
_C.OPT.TOTAL_EPOCHS = -1 # ensure this is integer valued, if -1 then train until train loss reaches epsilon
_C.OPT.TRAIN_LOSS_EPSILON_STOP = 1e-3
_C.OPT.GRAD_CLIP = None # set to None to disable

## Runtime Configurations
_C.SYSTEM.SEED = None #120899#120359
if torch.cuda.is_available():
    _C.SYSTEM.DEVICE = 'cuda'
else:
    _C.SYSTEM.DEVICE = 'cpu'

def get_cfg_defaults():
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
