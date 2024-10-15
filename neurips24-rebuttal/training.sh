# vary n
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 200
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 500
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 1000
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 2000

# vary n
python3 u=f.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 200
python3 u=f.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 500
python3 u=f.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 1000
python3 u=f.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder table_relu2 --batch_size 2000

# change act
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu  --model mlp --folder overparam_act  --batch_size 500
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu2  --model mlp --folder overparam_act --batch_size 500
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu25  --model mlp --folder overparam_act --batch_size 500
python3 poly_sin_gaussian.py --dim 2  --width 256  --id 0 --act relu3  --model mlp --folder overparam_act  --batch_size 500