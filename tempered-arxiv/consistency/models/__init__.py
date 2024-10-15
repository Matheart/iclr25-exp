import torch

from .feed_forward import FeedForwardNet, FeedForwardNTK
from .resnet import resnet18,resnet34,resnet18,resnet50,resnet101,resnet152
from .vit import vit4, vit8
from .wide_resnet import Wide_ResNet, conv_init

def get_model(cfg):
    if cfg.MODEL.MODEL_TYPE == 'feedforward':
        network = FeedForwardNet(cfg.MODEL.INP_DIM, cfg.MODEL.HID_DIM, cfg.MODEL.OUT_DIM, cfg.MODEL.N_HID_LAYERS, act_fn=cfg.MODEL.ACT_FN, fix_backbone=cfg.MODEL.FIX_BACKBONE)
    elif cfg.MODEL.MODEL_TYPE == 'feedforward-ntk':
        network = FeedForwardNTK(cfg.MODEL.INP_DIM, cfg.MODEL.HID_DIM, cfg.MODEL.OUT_DIM, cfg.MODEL.N_HID_LAYERS, act_fn=cfg.MODEL.ACT_FN, fix_backbone=cfg.MODEL.FIX_BACKBONE)
    elif cfg.MODEL.MODEL_TYPE == 'resnet18':
        network = resnet18(num_classes=cfg.MODEL.OUT_DIM, nchannels=cfg.MODEL.INP_DIM)
    elif cfg.MODEL.MODEL_TYPE == 'resnet34':
        network = resnet34(num_classes=cfg.MODEL.OUT_DIM, nchannels=cfg.MODEL.INP_DIM)
    elif cfg.MODEL.MODEL_TYPE == 'resnet50':
        network = resnet50(num_classes=cfg.MODEL.OUT_DIM, nchannels=cfg.MODEL.INP_DIM)
    elif cfg.MODEL.MODEL_TYPE == 'vit4':
        network = vit4(num_classes=cfg.MODEL.OUT_DIM, channels=cfg.MODEL.INP_DIM, image_size=cfg.DATA.IMAGE_SIZE)
    elif cfg.MODEL.MODEL_TYPE == 'vit8':
        network = vit8(num_classes=cfg.MODEL.OUT_DIM, channels=cfg.MODEL.INP_DIM, image_size=cfg.DATA.IMAGE_SIZE)
    elif cfg.MODEL.MODEL_TYPE.startswith('wide-resnet'):
        # expects model type to be given as string formatted like:
        # "wide-resnet {depth}x{k}"
        params = cfg.MODEL.MODEL_TYPE.strip().split(' ')[1].strip().split('x')
        depth = int(params[0])
        widen_factor = int(params[1])
        network = Wide_ResNet(depth, widen_factor, cfg.MODEL.DROPOUT_RATE,
                              cfg.MODEL.OUT_DIM, inp_channels=cfg.MODEL.INP_DIM)
    else:
        raise Exception('Unsupported model type: %s' % (cfg.MODEL.MODEL_TYPE))

    network = init_model(network, cfg)

    return network

def init_network(m, init_dict):
    if type(m) == torch.nn.Linear:
        if init_dict['type'] == 'normal':
            torch.nn.init.normal_(m.weight, mean=init_dict['mu'], \
                                  std=init_dict['std'])
            m.bias.data.fill_(init_dict['bias'])

def init_model(network, cfg):
    if cfg.MODEL.MODEL_TYPE.startswith('wide-resnet'):
        network.apply(conv_init)
    elif cfg.MODEL.INIT['type'] != 'default':
        print('normal init')
        network.apply(lambda module: init_network(module, cfg.MODEL.INIT))

    return network
