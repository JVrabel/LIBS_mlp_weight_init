import pickle
import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, input_shape: int, hidden_units1: int, hidden_units2: int, output_shape: int, weights_path=None):
        super().__init__()
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(input_shape, hidden_units1),
            nn.ReLU()
        )
        self.hidden_layer_2 = nn.Sequential(
            nn.Linear(hidden_units1, hidden_units2),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_units2, output_shape)
        
        # Load weights if a path is provided
        first_layer_weights = self.load_weights(weights_path) if weights_path else None
        
        # Initialize weights after model construction
        self.init_weights(first_layer_weights)

    def load_weights(self, weights_path):
        # Load the tensor from the .pkl file
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        return weights

    def init_weights(self, first_layer_weights=None):
        if first_layer_weights is not None:
            self.hidden_layer_1[0].weight = nn.Parameter(first_layer_weights)
            if self.hidden_layer_1[0].bias is not None:
                nn.init.constant_(self.hidden_layer_1[0].bias, 0)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.hidden_layer_1[0]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x
