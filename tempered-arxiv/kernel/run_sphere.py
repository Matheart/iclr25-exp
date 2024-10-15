'''Use case: MNIST'''

import mnist
import torch
from dsphere import load_sphere
import kernel
import eigenpro
import numpy as np
import ipdb
from utils import addlabelnoise
import csv
from datetime import datetime
time_stamp = datetime.today().strftime('%Y-%m-%d-%H:%M') 

subsamples = [100,500,1000,2000,5000,10000,50000,100000,500000,1000000,2000000]

bandwidth = 1
cutoff = 10**-4
weight_decay = None#0.00001
kernel_type = "gaussian" #"gaussian"#"laplacian"
dataset = "sphere"



dim=1000
if weight_decay==None:
    results_fp = open(f'{time_stamp}_{dim}-sphere_{kernel_type}_EIGENPRO.csv', 'w')
else:
    results_fp = open(f'{time_stamp}_{dim}-sphere_{kernel_type}_ridge{weight_decay}_EIGENPRO.csv', 'w')
    
results_writer = csv.writer(results_fp, delimiter=',', quotechar='"')
results_writer.writerow(["Kernel Type", "kernel_bandwidth","weight_decay","epochs","train_n" ,"test_n", 
                         "train_mse", 
                         "val_mse","seconds","dim"])
results_fp.flush()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(120)
np.random.seed(120)


######eigenpro
for s in subsamples:
    print(f'Number of Training samples:{s}')

    x_train, y_train, x_test, y_test = load_sphere(s,dim)
    n_class = y_train.shape[1]

    if kernel_type=="gaussian":
        kernel_fn = lambda x,y: kernel.gaussian(x, y, bandwidth=bandwidth)
    elif kernel_type=="laplacian":
        kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=bandwidth)

    model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
    res,alpha_eigp = model.fit(x_train, y_train, x_test, y_test,cutoff=cutoff, epochs=[1, 2, 5,100], mem_gb=12,weight_decay = weight_decay)

    epochs = [*res.keys()]
    last_epoch = epochs[-1]
    results_writer.writerow([kernel_type,bandwidth,weight_decay,last_epoch,y_train.shape[0],y_test.shape[0],
                             res[last_epoch][0]['mse'].item(),
                             res[last_epoch][1]['mse'].item(),res[last_epoch][2],dim])

    results_fp.flush()

