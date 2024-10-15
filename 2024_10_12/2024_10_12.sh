cd ..
# 3 x 512
# vary noise level
# 5e-2, 1e-1, 5e-1, 1 
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