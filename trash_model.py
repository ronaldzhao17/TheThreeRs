import torch
import torch.nn as nn
from torchvision import models
import multiprocessing

# Only print in the main process
if multiprocessing.current_process().name == "MainProcess":
    print("loaded model")

class TrashModel(nn.Module):
    def __init__(self, num_classes):
        super(TrashModel, self).__init__()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Replace the final fully-connected layer with one that outputs num_classes scores
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

