"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="xxx.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)




def resample_spectra_df(df, original_wavelengths, new_wavelengths):
    """
    Resample the spectra in dataset X (as a DataFrame) to new wavelengths using linear interpolation.

    Parameters:
    - df: DataFrame, where each row is a spectrum.
    - original_wavelengths: 1D array of original wavelengths.
    - new_wavelengths: 1D array of new wavelengths.

    Returns:
    - df_resampled: DataFrame of resampled spectra.
    """

    # Convert DataFrame to NumPy array
    X = df.values

    num_spectra = X.shape[0]
    X_resampled = np.zeros((num_spectra, len(new_wavelengths)))

    for i in range(num_spectra):
        f = interp1d(original_wavelengths, X[i, :], kind='linear', fill_value='extrapolate')
        X_resampled[i, :] = f(new_wavelengths)

    # Convert back to DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=new_wavelengths)

    return df_resampled


def compute_sparsity(model, threshold=1e-10):
    """
    Compute the sparsity of the model and each layer, focusing on weights.
    
    Args:
    - model: PyTorch model.
    - threshold: float, weights with absolute value below this are considered zero.
    
    Returns:
    - total_sparsity: float, the overall sparsity of the model.
    - layer_sparsities: dict, sparsity of each layer.
    - first_hidden_layer_sparsity: float, sparsity of the first hidden layer.
    """
    total_zero_weights = 0
    total_weights = 0
    layer_sparsities = {}
    first_hidden_layer_sparsity = None  # Initialize to None, will update if found

    for name, param in model.named_parameters():
        if "weight" in name:  # Focus on weight parameters
            tensor = param.data
            total = tensor.numel()
            zero_weights = (tensor.abs() < threshold).sum().item()
            
            sparsity = zero_weights / total
            layer_sparsities[name] = sparsity
            
            total_zero_weights += zero_weights
            total_weights += total

            # Check if this is the first hidden layer by name
            if name == 'hidden_layer_1.0.weight':  # Adjust this if your layer naming is different
                first_hidden_layer_sparsity = sparsity

    total_sparsity = total_zero_weights / total_weights
    
    # Ensure the first hidden layer was found; otherwise, report it was not found
    if first_hidden_layer_sparsity is None:
        first_hidden_layer_sparsity = 'First hidden layer not found. Please check the layer names.'
    
    return total_sparsity, layer_sparsities, first_hidden_layer_sparsity

# # Example usage:
# threshold = 1e-10  # Adjust the threshold as needed
# total_sparsity, layer_sparsities, first_layer_sparsity = compute_sparsity(model, threshold)

# print(f"Total model sparsity: {total_sparsity:.2%}")
# print(f"First hidden layer sparsity: {first_layer_sparsity if isinstance(first_layer_sparsity, str) else f'{first_layer_sparsity:.2%}'}")


def sigmoid_schedule(epoch,reg_lambda, x_0, steepness = 0.0125):

    """
    Generic sigmoid schedule with definable x_0, steepness, clipped between [0, reg_lambda].
    
    Args:
    - epoch (int): Current epoch.
    - x_0 (int): The x-value (epoch) at which the sigmoid function is centered.
    - steepness (float): Controls the steepness of the sigmoid curve.
    - reg_lambda (float): The maximum value for the sigmoid, defining the upper clip limit.
    
    Returns:
    - float: The value of the sigmoid function for the given epoch, clipped between 0 and reg_lambda.
    """
    # Sigmoid function
    value = reg_lambda / (1 + np.exp(-steepness * (epoch - x_0)))
    return value
# epochs = np.arange(0, 1024)
# sigmoid_values = [generic_sigmoid_schedule(epoch, x_0 = 512, steepness = 0.02, reg_lambda = 2) for epoch in epochs]
# plt.plot(sigmoid_values)