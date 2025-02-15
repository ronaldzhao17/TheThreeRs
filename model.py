import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#from dataset import CLASSES

# Define number of classes based on the dataset
# num_classes = len(CLASSES)  # Get number of categories from dataset
num_classes = 16

# Load a pretrained ResNet-34 model
model = models.resnet34(pretrained=True)

# Modify the final classification layer to match the number of trash categories
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()  # Standard loss for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Print Model Summary
print(model)
