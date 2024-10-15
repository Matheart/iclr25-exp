import os
import argparse
import math
import numpy as np
import numpy.random as npr
from numpy import linalg
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset
import torch.optim as optim
import time
from math import *
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import wandb

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type = int, default = 2)
parser.add_argument('--width', type = int, default = 256)
parser.add_argument('--num_layer', type = int, default = 2)
#parser.add_argument('--id', type = int, default = 0)
parser.add_argument('--relu_power', type = int, default = 1)
#parser.add_argument('--model', type = str, default = 'mlp')
parser.add_argument('--test', type = str, default = 'false')
parser.add_argument('--folder', type = str, default = 'test') # experiment folder
parser.add_argument('--sample_size', type = int, default = 500)
parser.add_argument('--noise_level', type = float, default = 1e-2)
parser.add_argument('--inv_op_power', type = int, default = 0) # default: identity, 1: Delta, 2: Delta^2
args = parser.parse_args()

args.test = True if args.test == 'true' else False
inv_op_power = args.inv_op_power

# Generate output file name
def generate_output_filename(args):
    return f"log/{args.folder}/output_dim{args.dim}_width{args.width}_layers{args.num_layer}_relu{args.relu_power}_size{args.sample_size}_noise{args.noise_level:.2e}_invop{args.inv_op_power}"

save_path = generate_output_filename(args)

class ReLUPower(nn.Module):
    def __init__(self, power):
        super(ReLUPower, self).__init__()
        self.power = power

    def forward(self, x):
        return torch.relu(x) ** self.power

class MLP(nn.Module): # Three-layer MLP
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, activation):
        super(MLP, self).__init__()
        # input_dim: 2
        # hidden_dim: 256
        # output_dim: 1
        self.num_layer = num_layer

        hidden_layers = []
        for i in range(num_layer):
            hidden_layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.hidden_layers = torch.nn.Sequential(*hidden_layers)

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        for fc in self.hidden_layers:
            # print(x.device)
            # k = fc(x)
            # print(k.device)
            x = self.activation(fc(x))
        x = self.fc_out(x)
        return x

# def operator(f, inv_op_power = 0):
#     if inv_op_power == 0:
#         return f
#     else: # Laplacian
#         pass
#     #else: # Laplacian^2
#     #    pass

# x: 2D input
# sin(pi (x+y))
def ground_truth(x, inv_op_power):
    assert(x.shape == (args.sample_size, args.dim))
    x = x.T
    assert(x.shape == (args.dim, args.sample_size))
    if x.shape[0] == 2:
        y = torch.sin(np.pi * (x[0] + x[1]))
        #y = torch.ones_like(y)
        y = x[0] + x[1]
    else:
        raise NotImplementedError
        y = torch.sin(np.pi * (x[0]))
        #y = torch.ones_like(y)
        #y = x[0] + x[1]
    if inv_op_power == 1:
        # pi cos(pi(x+y))
        # -pi^2 sin(pi(x+y))
        y *= (- torch.pi ** 2)
    elif inv_op_power == 2:
        raise NotImplementedError
        
    assert(y.shape == (args.sample_size, ))
    return y

def set_sweep_config():
    sweep_config = {
        'method': 'random'
    }

    metric = {
        'name': 'loss',
        'goal': 'minimize'   
    }

    sweep_config['metric'] = metric
    parameters_dict = {
        'fc_layer_size': {
            'values': [128, 256, 512, 1024]
        },
        'lr': {
            'values': [5e-5, 1e-5, 5e-4, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        },
        'num_layer': {
            'values': [2, 3]
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    return sweep_config, sweep_id

wandb.init(config = None)
config = wandb.config

act = ReLUPower(args.relu_power)
model = MLP(
    input_dim = args.dim, 
    hidden_dim = config.fc_layer_size, 
    output_dim = 1, 
    num_layer = config.num_layer,
    activation = act
).to(device)
#model.apply(lambda m: nn.init.kaiming_normal_(m.weight) if isinstance(m, nn.Linear) else None)
epoch = 30000 # 30000

# need to tune the weight
optimizer = optim.Adam([
                {'params': model.parameters()}
            ], lr = config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay= 1e-4) # need to tune
# 5e-5
scheduler =  StepLR(optimizer, step_size = 2000, gamma=0.7)

# generate dataset
train_x = torch.rand(args.sample_size, args.dim).to(device)
train_y = ground_truth(train_x, inv_op_power).to(device) +  (args.noise_level ** 0.5) * torch.randn(train_x.size()[0], ).to(device) # ground truth: 0
#ground_truth(train_x).to(device) 

#print(train_y.shape)
assert(train_y.shape == (args.sample_size, ))
test_x  = torch.rand(args.sample_size, args.dim).to(device)
test_y  = ground_truth(test_x, inv_op_power).to(device) # clean testing
assert(test_y.shape == (args.sample_size, ))

# begin training
time_start = time.time()

# satisfy boundary condition on [0,1]^2
def model_with_boundary(x):
    return torch.prod(x*(1-x),dim=1).reshape([x.size()[0],1])*model(x) 

def compute_loss(train_x, train_y, inv_op_power):
    assert(train_x.shape == (args.sample_size, args.dim))
    #print(train_y.shape)
    assert(train_y.shape == (args.sample_size, ))

    if inv_op_power != 0:
        train_x.requires_grad = True

    predict_y = model_with_boundary(train_x)
    train_y = train_y.reshape(-1, 1)
    #print('predict_y:', predict_y.shape) # (500, 1)

    assert(train_y.shape == (args.sample_size, 1))
    assert(predict_y.shape == (args.sample_size, 1))

    if inv_op_power == 0:
        losses = torch.sum((predict_y - train_y) ** 2) / args.sample_size
    elif inv_op_power == 1: # -u = Delta f, contain bug
        v   = torch.ones(predict_y.shape).to(device)
        ux  = torch.autograd.grad(predict_y, train_x,grad_outputs=v,create_graph=True)[0]
        uxx = torch.zeros(args.sample_size, args.dim).to(device)

        for i in range(args.dim):
            ux_tem = ux[:,i].reshape([args.sample_size, 1])
            #print('ux_tem:', ux_tem)
            uxx_tem = torch.autograd.grad(ux_tem,train_x,grad_outputs=v,create_graph=True)[0]
            #print('uxx_tem:', uxx_tem)
            uxx[:,i] = uxx_tem[:,i]

        losses = torch.sum((train_y - torch.sum(uxx, dim = 1).reshape([args.sample_size, 1])) ** 2) / args.sample_size
    return losses

clean_test_loss_record = []

# train to overfit
for i in tqdm(range(epoch)):
    optimizer.zero_grad()
    model.train()
    losses = compute_loss(train_x, train_y, inv_op_power)
    losses.backward()
    optimizer.step() 
    # error = loss_error()
    # error_save[i]=float(error)
    model.eval()
    test_losses = compute_loss(test_x, test_y, inv_op_power)

    scheduler.step()

    if i % 100 == 0:
        print('epoch', i, ' training loss:', losses.item())
        print('epoch', i, ' clean testing loss:', test_losses.item())
        # log clean testing loss
        if not args.test:
            clean_test_loss_record.append(test_losses.item())

        print()

    # early stops == regularized, don't include it
    # if losses < 1e-6: 
    #     print('epoch', i, 'training loss < 1e-4, early stopping!')
    #     break

if not args.test:
    print('Saving the clean test loss file into', save_path ,'...')
    np.save(save_path, clean_test_loss_record)


time_end = time.time()
print('time cost', time_end - time_start, 's')

