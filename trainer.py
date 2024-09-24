# Define trainer class
import torch
from torch import optim, nn

class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    # Function that trains the model
    def train(self, dataloader):


