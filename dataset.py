import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import multiprocessing

# Only print in the main process
if multiprocessing.current_process().name == "MainProcess":
    print("loaded dataset")

# Define transforms for training and testing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset
dataset = datasets.ImageFolder(root="data", transform=train_transforms)

# Save the number of classes based on the dataset
num_classes = len(dataset.classes)

# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# For test_dataset, use the test_transforms
test_dataset.dataset.transform = test_transforms

# Create DataLoaders (set num_workers=0 to avoid extra process spawns if desired)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
