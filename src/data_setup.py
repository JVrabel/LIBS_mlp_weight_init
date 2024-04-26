"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from load_libs_data import load_contest_train_dataset, load_contest_test_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    batch_size: int, 
    num_classes: int,
    device: torch.device,
    num_workers: int=NUM_WORKERS, 
    split_rate: float=0.3,
    random_st: int=102,
    spectra_count: int=100
):
    """Creates training and validation DataLoaders.
    ...
    """
    
    pickle_file_path = "data/data.pkl"
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data_dict = pickle.load(f)
        X = data_dict['X']
        y = data_dict['y']
        samples = data_dict['samples']
    else:
        X, y, samples = load_contest_train_dataset(train_dir, spectra_count)
        with open(pickle_file_path, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'samples': samples}, f)
        
    wavelengths = X.columns

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_rate, random_state=random_st, stratify=samples, shuffle=True)
    del y, samples, X
    
    y_train = y_train-1
    y_val = y_val-1
    
    scaler = Normalizer(norm='max')
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_val = torch.from_numpy(np.array(y_val)).long()
    
    # X_train = X_train.to(device)
    # X_val = X_val.to(device)
    # y_train = y_train.to(device)
    # y_val = y_val.to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, y_train



def create_test_dataloader(
    test_dir: str, 
    test_labels_dir: str, 
    batch_size: int, 
    device: torch.device
    ):
    """Creates the test DataLoader.

    """
    X_test = load_contest_test_dataset(test_dir)
    y_test = np.loadtxt(test_labels_dir, delimiter = ',')


    if True:
      scaler =  Normalizer(norm='max')
      X_test = scaler.fit_transform(X_test)

    # Convert data to torch tensors
    X_test = torch.from_numpy(X_test).float() # Add extra dimension for channels
    y_test = torch.from_numpy(np.array(y_test)).long()

    # If available, move data to the GPU
    # X_test.to(device)
    # y_test.to(device)
    
    # Create PyTorch DataLoader objects for the training and validation sets
    test_dataset = TensorDataset(X_test, y_test)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader, y_test
