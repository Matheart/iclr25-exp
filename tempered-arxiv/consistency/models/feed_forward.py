import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class FeedForwardNet(torch.nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, n_hidden_layers, act_fn='relu', fix_backbone=False):
        super(FeedForwardNet, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        hidden_layers = []
        for idx in range(n_hidden_layers):
            if idx == 0:
                hid = torch.nn.Linear(inp_dim, hid_dim, bias=True)
            else:
                hid = torch.nn.Linear(hid_dim, hid_dim, bias=True)

            if fix_backbone:
                hid.requires_grad = False

            hidden_layers.append(hid)

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_layer = torch.nn.Linear(hid_dim, out_dim, bias=True)

        if act_fn == 'relu':
            self.act_fn = F.relu
        elif act_fn == 'erf':
            self.act_fn = torch.special.erf
        elif act_fn == 'leaky_relu':
            self.act_fn = F.leaky_relu
        else:
            raise ValueError(f'act_fn: {act_fn} not supported')

    def forward(self, x):
        for idx, hid in enumerate(self.hidden_layers):
            x = self.act_fn(hid(x))
        return self.output_layer(x)

class FeedForwardNTK(torch.nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim, n_hidden_layers, act_fn='relu', fix_backbone=False):
        super(FeedForwardNTK, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        hidden_layers = []
        for idx in range(n_hidden_layers):
            if idx == 0:
                hid = torch.nn.Linear(inp_dim, hid_dim, bias=True)
            else:
                hid = torch.nn.Linear(hid_dim, hid_dim, bias=True)

            if fix_backbone:
                hid.requires_grad = False

            hidden_layers.append(hid)

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)
        self.output_layer = torch.nn.Linear(hid_dim, out_dim, bias=True)

        if act_fn == 'relu':
            self.act_fn = F.relu
        elif act_fn == 'erf':
            self.act_fn = torch.special.erf
        elif act_fn == 'leaky_relu':
            self.act_fn = F.leaky_relu
        else:
            raise ValueError(f'act_fn: {act_fn} not supported')

    def forward(self, x):
        for idx, hid in enumerate(self.hidden_layers):
            x = self.act_fn(1/np.sqrt(self.hid_dim)*hid(x))
        return 1/np.sqrt(self.out_dim)*self.output_layer(x)
