''Exact solutiont to Kernel Machine for d-sphere synthetic data''


from dsphere import load_sphere 
import kernel
import torch
import numpy as np
import csv
from datetime import datetime

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--kernel_type', type=str, nargs='+')
parser.add_argument('--ridge', type=float)


time_stamp = datetime.today().strftime('%Y-%m-%d-%H:%M') 


def run_mse(dims,bandwidth,n_power=14,repeat=1,ridge=0,kernel_type="Gaussian"):
    ##### Model 
    if kernel_type == "Gaussian":
        kernel_fn  = lambda x,y: kernel.gaussian(x, y, bandwidth=bandwidth)
    elif kernel_type == "Laplacian":
        kernel_fn = lambda x,y: kernel.laplacian(x, y, bandwidth=bandwidth)



    mse_test_gaussian = []
    mse_test_laplacian = []

    ns = [2**i for i in range(3,n_power)]



    results_fp = open(f'{time_stamp}-sphere_{kernel_type}_ridge{ridge}_exact.csv', 'w')    
    results_writer = csv.writer(results_fp, delimiter=',', quotechar='"')
    results_writer.writerow(["Kernel Type", "kernel_bandwidth","ridge","train_n","val_mse","dimension"])
    results_fp.flush()
    for d in dims:
        print(f'dimension={d}')
        for s_ind,s in enumerate(ns):
            loss = 0
            for r in range(repeat):
                x_train, y_train, x_test, y_test = load_sphere(s,d)
                x_train = torch.tensor(x_train)
                x_test = torch.tensor(x_test)
                Kxx_train = kernel_fn(x_train,x_train)
                Kxx_test = kernel_fn(x_test,x_train)
                alpha_opt = torch.inverse(Kxx_train+ridge*torch.eye(x_train.shape[0]))@y_train.float()
                loss_tmp = torch.sum( torch.pow( Kxx_test@alpha_opt - y_test , 2) )/y_test.shape[0]
                results_writer.writerow([kernel_type,bandwidth,ridge,y_train.shape[0],loss_tmp.item(),d])
                results_fp.flush()
                
if __name__ == "__main__":
    
    args = parser.parse_args()
    dims = [5,10,15]
    bandwidth = 5
    run_mse(dims,bandwidth,n_power=14,repeat=100,ridge=args.ridge,kernel_type=args.kernel_type[0])
    
    
    
    
    
    
                