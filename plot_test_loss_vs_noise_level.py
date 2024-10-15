import numpy as np
import matplotlib.pyplot as plt

# Define the noise levels
noise_levels = [5e-2, 1e-1, 5e-1, 1]#[5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50]

# regression, pinn + relu, pinn + relu2, pinn + sin
case_labels = ['regression', 'pinn + relu', 'pinn + relu2', 'pinn + sin']
case_errors = []
colors = ['red', 'orange', 'blue', 'purple']

# Load data for each noise level for invop0
for case_label in case_labels:
    case_error = []
    if 'regression' in case_label:
        op_num = 0
        lr = 5e-3
    else:
        op_num = 1
        lr = 5e-5

    if 'relu2' in case_label:
        act = 'relu2'
    elif 'sin' in case_label:
        act = 'sin'
    else:
        act = 'relu'

    for noise in noise_levels:
        file_path = f"/home/howon/iclr25-exp/log/new_plot/output_dim2_width1024_layers2_{act}_size500_noise{noise:.2e}_invop{op_num}_lr{lr}_train.npy"
        data = np.load(file_path, allow_pickle=True)
        case_error.append(data[-1])
    case_errors.append(case_error)

# Create the plot
plt.figure(figsize=(12, 8))

for index, (case_label, case_error) in enumerate(zip(case_labels, case_errors)):
    plt.loglog(noise_levels, case_error, marker='o', color = colors[index], label=case_label)

# Plot y = x line
plt.loglog(noise_levels, noise_levels, linestyle='--', color='green', label='y = x')

# Beautify the plot
plt.xlabel('Noise Variance', fontsize=14)
plt.ylabel('Clean Test Error', fontsize=14)
plt.title('Benign Overfitting: Train Error vs Noise Level Variance', fontsize=16)
plt.legend(loc='best', fontsize=12)  # Added legend here
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(noise_levels, [f"{nl:.0e}" for nl in noise_levels], fontsize=12)
plt.yticks(fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig('2024_10_13_train_loss_noise_level.png')

# Optionally display the plot
# plt.show()