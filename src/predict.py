import torch
import argparse
from sklearn.metrics import accuracy_score
from data_setup import create_test_dataloader
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='Testing script for PyTorch model.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')


args = parser.parse_args()


# Setup directories
test_labels_dir = "data/test_labels.csv"
#model_dir = 'models/L1_lambda_0.002_model.pth'
test_dir = "data/contest_TEST.h5"


# Setup target device 
#device = torch.device("cpu")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the model class from your specific location.
from model_builder import SimpleMLP

# Hyperparameters (same as train.py)
INPUT_SHAPE = 13750
OUTPUT_SHAPE = 12
HIDDEN_UNITS1 = 125  # Number of neurons in the first hidden layer
HIDDEN_UNITS2 = 128  # Number of neurons in the second hidden layer
BATCH_SIZE = 128


# Initialize the model
model = SimpleMLP(
    input_shape=INPUT_SHAPE,
    hidden_units1=HIDDEN_UNITS1,
    hidden_units2=HIDDEN_UNITS2,
    output_shape=OUTPUT_SHAPE
).to(device)

# Load the saved state_dict into the model
model_path = args.model_path
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode (important for layers like dropout, batchnorm, etc.)
model.eval()



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
predicted_classes = predicted_classes # Remove the singleton dimension, shape [N]

# Convert to numpy array for further processing
predicted_classes = predicted_classes.numpy()

print("Predicted classes:", predicted_classes)



y_test = y_test-1

# Compute accuracy
acc = accuracy_score(predicted_classes, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")



