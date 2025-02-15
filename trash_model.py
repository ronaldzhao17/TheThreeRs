import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#from dataset import CLASSES

# Define number of classes based on the dataset
# num_classes = len(CLASSES)  # Get number of categories from dataset
num_classes = 16

class TrashModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model.forward(x)


# Print Model Summary
print("loaded model")
