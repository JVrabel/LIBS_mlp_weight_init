import numpy as np

import subprocess

# List of lambda values you want to experiment with

# lambda_values = [1.6, 1.7, 1.8, 1.9, 2.0]
# lambda_values = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
# Define a finer range of lambda values around the promising region
# lambda_values = np.linspace(0.01, 2, num=20)  # Adjust the range and num based on coarse search results

start = np.log10(0.001)  # This is -2 because 10^-2 = 0.01
stop = np.log10(10)      # This is approximately log10(4)

# Generate values using logspace
num_values = 10  # You can adjust this number based on how many values you want
lambda_values = np.logspace(start, stop, num=num_values)

# lambda_values = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
# lambda_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] #, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
# lambda_values = [1,2,3,4] #, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

# Number of trials for each lambda value
num_trials = 2  # Feel free to change this

# Dictionary to hold metrics for each lambda value
all_metrics = {}

# Run the training script with each lambda value
for lambda_val in lambda_values:
    for trial_num in range(1, num_trials + 1):  # trial_num starts from 1
        # Execute the training script, passing in lambda and trial number as command line arguments
        subprocess.run(['python', 
                        'src/train.py', 
                        '--save_model', 'True', 
                        '--trial', str(trial_num),
                        '--epochs', '512'
                        ])
        