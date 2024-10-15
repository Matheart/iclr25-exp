#!/bin/bash 
#SBATCH --job-name=exp2
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
# pinn + sin + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act sin --sample_size 50 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act sin --sample_size 100 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act sin --sample_size 500 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act sin --sample_size 5000 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act sin --sample_size 10000 
# pinn + relu2 + adam
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu2 --sample_size 50 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu2 --sample_size 100 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu2 --sample_size 500 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu2 --sample_size 5000 
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 0.1 --num_layer 2  --width 1024 --act relu2 --sample_size 10000 