import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations with data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match common model input size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Rotate by ±10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Change brightness, contrast, etc.
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Define simpler transformations for testing
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = datasets.ImageFolder(root="data", transform=train_transforms)

# Split into training (80%) and testing (20%) sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply test transforms separately
test_dataset.dataset.transform = test_transforms

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Print some dataset details
print(f"Number of classes: {len(dataset.classes)}")
print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

# Display class names
print("Class names:", dataset.classes)

#CLASSES = dataset.classes
