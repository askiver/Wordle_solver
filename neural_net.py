# File for defining the neural network

import torch
import torch.nn as nn
import torch.nn.functional as F


# Class for the neural network
class NeuralNet(nn.Module):
    def __init__(self, num_layers: int, layer_breadth: int):
        super(NeuralNet, self).__init__()
        layers = []
        # Create the input layer
        layers.append(nn.Linear(26*5*3, layer_breadth))
        # Create the hidden layers
        for i in range(num_layers-1):
            layers.append(nn.Linear(layer_breadth, layer_breadth))
        # Create the output layer
        layers.append(nn.Linear(layer_breadth, 26*5))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.GELU(layer(x))
        x = self.layers[-1](x)
        # Reshape the output for softmax
        x = x.view(-1, 26, 5)

        x = torch.softmax(x, dim=1)
        return x

    def get_guess(self, x):
        # Get the guess from the model
        guess = self.forward(x)
        # change the values to 0 or 1
        max_indices = torch.argmax(guess, dim=0)
        one_hot_output = torch.nn.functional.one_hot(max_indices, num_classes=26).T
        return one_hot_output.flatten()
