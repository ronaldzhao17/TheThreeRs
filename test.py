# test.py
import torch
from PIL import Image
from torchvision import transforms 
from trash_model import TrashModel
from dataset import dataset, num_classes
# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load saved weights
model = TrashModel(num_classes).to(device)
model.load_state_dict(torch.load("trash_model.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Define the same transform used during training/inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load a test image (replace 'test.jpg' with your image file)
test_image_path = "image.jpg"
try:
    image = Image.open(test_image_path).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Apply transform and add batch dimension
input_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    print("start")
    output = model(input_tensor)
    print("stop")
    predicted_class = output.argmax(dim=1).item()
    trash_type = dataset.classes[predicted_class]

print(f"Trash type: {trash_type}")
