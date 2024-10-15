#!/bin/bash 
#SBATCH --job-name=plot
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

python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 100
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 1000
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 5000
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10000