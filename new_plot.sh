5e-3 interpolation threshold
5e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1


1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1

1e-1 => 2e-2 interpolation
5e-2
1e-2

1e-1 3e-1 5e-1 7e-1 1e0


# target: make regresssion problem loss drops to 1e-3
# noise level 1e-1
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 3e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 5e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 7e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e0 --num_layer 3  --width 512 

python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 3e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 5e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 7e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e0 --num_layer 3  --width 512 --act sin

python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --act sin
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 3e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 5e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 7e-1 --num_layer 3  --width 512 --act sin
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e0 --num_layer 3  --width 512 --act sin

# Delta^2
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 3e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 5e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 7e-1 --num_layer 3  --width 512 
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e0 --num_layer 3  --width 512

# keep noise level 1e-1, plot testing loss vs sample size
10, 50, 100, 1000, 5000, 10000
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 50
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 100
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 1000
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 5000
python nn_exp.py --inv_op_power 0 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10000

python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 50
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 100
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 1000
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 5000
python nn_exp.py --inv_op_power 1 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10000

python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 50
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 100
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 1000
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 5000
python nn_exp.py --inv_op_power 2 --folder new_plot --noise_level 1e-1 --num_layer 3  --width 512  --sample_size 10000
