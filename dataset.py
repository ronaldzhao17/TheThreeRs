import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import multiprocessing

# Only print in the main process
if multiprocessing.current_process().name == "MainProcess":
    print("loaded dataset")

# Define transforms for training and testing with additional augmentations for training
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),                   # Resize to a larger size first
    transforms.RandomResizedCrop(224),               # Randomly crop to 224x224
    transforms.RandomHorizontalFlip(),               # Randomly flip horizontally
    transforms.RandomRotation(15),                   # Increase rotation range to 15 degrees
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Adjust brightness, contrast, etc.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    transforms.ToTensor(),                           # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))  # Randomly erase parts of the image
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the dataset with the training transforms
dataset = datasets.ImageFolder(root="data", transform=train_transforms)

# Save the number of classes based on the dataset
classes = dataset.classes
num_classes = len(classes)

# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# For test_dataset, use the test_transforms
test_dataset.dataset.transform = test_transforms

# Create DataLoaders (set num_workers=0 to avoid extra process spawns if desired)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
