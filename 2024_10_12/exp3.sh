#!/bin/bash 
#SBATCH --job-name=exp3
#SBATCH --mail-user=howon@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=48:00:00
#SBATCH --account=vvh
#SBATCH --partition=vvh-l40s
#SBATCH --gpus=l40s
#SBATCH --output=/home/%u/%x-%j.log

cd ..
# pinn + relu2 + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-2 --num_layer 2  --width 1024 --act relu2 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 2  --width 1024 --act relu2
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-1 --num_layer 2  --width 1024 --act relu2
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1 --num_layer 2  --width 1024 --act relu2
# pinn + relu + lbfgs
# pinn + sin + lbfgs

# vary sample size , fix noise = 0.1
# 50, 100, 500, 5000, 10000
# regression + relu
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 50
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 100
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 500
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 5000
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 10000
# pinn + relu + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 50
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 100
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 500
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 5000
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu --sample_size 10000