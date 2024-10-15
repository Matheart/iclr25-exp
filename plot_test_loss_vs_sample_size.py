import numpy as np
import matplotlib.pyplot as plt

# Fix the noise level
fixed_noise_level = 1e-1

# Define the sample sizes to vary
sample_sizes = [50, 100, 500, 5000, 10000]
clean_test_errors = []

# regression, pinn + relu, pinn + relu2, pinn + sin
case_labels = ['regression', 'pinn + relu', 'pinn + relu2', 'pinn + sin']
case_errors = []
colors = ['red', 'orange', 'blue', 'purple']

# Load data for each noise level for invop0
for case_label in case_labels:
    case_error = []
    if 'regression' in case_label:
        op_num = 0
        lr = "0.005"
    else:
        op_num = 1
        lr = "5e-05"

    if 'relu2' in case_label:
        act = 'relu2'
    elif 'sin' in case_label:
        act = 'sin'
    else:
        act = 'relu'
    
    for n in sample_sizes:
        file_path = f"/home/howon/iclr25-exp/log/new_plot/output_dim2_width1024_layers2_{act}_size{n}_noise1.00e-01_invop{op_num}_lr{lr}_train.npy"
        data = np.load(file_path, allow_pickle=True)
        case_error.append(data[-1])
    case_errors.append(case_error)

# Create the plot
plt.figure(figsize=(12, 8))
for index, (case_label, case_error) in enumerate(zip(case_labels, case_errors)):
    plt.loglog(sample_sizes, case_error, marker='o', color = colors[index], label=case_label)
plt.legend(loc='best', fontsize=12)
plt.xlabel('Sample Size')
plt.ylabel('Training Error')
plt.title(f'Benign Overfitting: Training Error vs Sample Size (Noise Level = {fixed_noise_level:.2e})')
plt.grid(True)
plt.xticks(sample_sizes)  # Set x-ticks to the sample sizes
plt.savefig('2024_10_13_train_loss_vs_sample_size.png')
#plt.show()  # Display the plot
