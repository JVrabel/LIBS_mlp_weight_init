"""
Trains a PyTorch model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import win32com.client
from data_setup import create_test_dataloader
from sklearn.metrics import accuracy_score
import argparse
import csv

# Argument parser setup
parser = argparse.ArgumentParser(description='Training script for PyTorch model.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to run')
parser.add_argument('--save_model', type=bool, default=False, help='Flag to indicate if the model should be saved')
parser.add_argument('--trials', type=int, default=1, help='Number of trials for each setup')

args = parser.parse_args()

# Then use args to set your hyperparameters
NUM_EPOCHS = args.epochs
save_model = args.save_model
trials = args.trials




# Setup hyperparameters
# NUM_EPOCHS = 300
# regularization_type = "L1" # "vanilla" "sparseloc"
# reg_lambda = 0.1
BATCH_SIZE = 128
INPUT_SHAPE = 40000  # Modify this based on your actual input vector length
OUTPUT_SHAPE = 12
HIDDEN_UNITS1 = 128  # Number of neurons in the first hidden layer
HIDDEN_UNITS2 = 128  # Number of neurons in the second hidden layer
LEARNING_RATE = 0.0001


PROJECT_WANDB = 'LIBS_weight_init'
ENTITY_WANDB = 'jakubv'

# Setup directories for data - modify these paths as needed
WEIGHTS_PATH = 'data/first_layer_weights.pkl'
shell = win32com.client.Dispatch("WScript.Shell")
shortcut = shell.CreateShortCut('data/contest_TRAIN.h5.lnk')
train_dir = shortcut.Targetpath



# train_dir = "data/train"
# test_dir = "data/test"   # this should be val, and also used only if there is a specific dataset for valiadation data. 

# Setup target device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(
    train_dir=train_dir,
    batch_size=BATCH_SIZE,
    device = device,
    num_classes = OUTPUT_SHAPE
)

# Create model with help from model_builder.py
model = model_builder.SimpleMLP(
    input_shape=INPUT_SHAPE,
    hidden_units1=HIDDEN_UNITS1,
    hidden_units2=HIDDEN_UNITS2,
    output_shape=OUTPUT_SHAPE,
    weights_path=WEIGHTS_PATH
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             trial_num=trials,
             project_wandb=PROJECT_WANDB,
             entity_wandb=ENTITY_WANDB)


# Save the model with help from utils.py
if save_model:
    model_name = f"{LEARNING_RATE}_model.pth"
    utils.save_model(model=model,
                     target_dir="models",
                     model_name=model_name)


device = torch.device('cpu')
model.to(device)

test_labels_dir = "data/test_labels.csv"
#model_dir = 'models/L1_lambda_0.002_model.pth'
test_dir = "data/contest_TEST.h5"
test_dataloader, y_test = create_test_dataloader(test_dir=test_dir,
                                                test_labels_dir=test_labels_dir,
                                                batch_size=BATCH_SIZE,
                                                device = device
                                                )



all_outputs = []  # To store all model outputs

with torch.no_grad():  # Deactivates autograd, reduces memory usage and speeds up computations
    for i, (input_data, labels) in enumerate(test_dataloader):
        output = model(input_data.to(device))  # Assuming output shape is [128, 1, 12]
        all_outputs.append(output.cpu())  # Move to CPU and store



# Concatenate all outputs along the batch dimension
all_outputs = torch.cat(all_outputs, dim=0)  # New shape will be [N, 1, 12] where N is the total number of samples

# Apply argmax to the last dimension to find the class with maximum probability
predicted_classes = torch.argmax(all_outputs, dim=-1)  # Shape will be [N, 1]
predicted_classes = predicted_classes  # Remove the singleton dimension, shape [N]

# Convert to numpy array for further processing
predicted_classes = predicted_classes.numpy()

print("Predicted classes:", predicted_classes)


y_test = y_test-1

# Compute accuracy
acc = accuracy_score(predicted_classes, y_test)


# Define the path for your results file
results_file_path = "results.csv"

# Check if the file exists to determine if headers need to be written
file_exists = os.path.exists(results_file_path)

# Open the file in append mode
with open(results_file_path, 'a', newline='') as csvfile:
    fieldnames = ['trials', 'test_acc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If the file did not exist, write the header
    if not file_exists:
        writer.writeheader()

    # Write the data
    writer.writerow({'trials': trials, 'test_acc': acc})

print(f"Test Accuracy: {acc * 100:.2f}% stored in {results_file_path}")