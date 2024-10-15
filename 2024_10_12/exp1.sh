#!/bin/bash 
#SBATCH --job-name=exp1
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
# regression + relu
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 5e-2 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 5e-1 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1 --num_layer 2  --width 1024 --act relu 
# pinn + relu + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-2 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-1 --num_layer 2  --width 1024 --act relu 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1 --num_layer 2  --width 1024 --act relu 
# pinn + sin + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-2 --num_layer 2  --width 1024 --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 2  --width 1024 --act sin 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-1 --num_layer 2  --width 1024 --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1 --num_layer 2  --width 1024 --act sin 