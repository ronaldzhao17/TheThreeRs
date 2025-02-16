# app.py
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Updated list of trash classes (order must match the model's output order)
trash_types = [
    'Brown box', 'Chopsticks', 'Foil', 'Food', 'Paper', 'Paper Bowl',
    'Plastic Container', 'Plastic Packets', 'Plastic Utensil', 'Plastic Wrap', 'Straw'
]

# Updated mapping from trash type to disposal bin
trash_to_bin = {
    'Brown box': 'recycling',
    'Chopsticks': 'recycling',
    'Foil': 'landfill',
    'Food': 'organic material',
    'Paper': 'recycling',
    'Paper Bowl': 'recycling',
    'Plastic Container': 'recycling',
    'Plastic Packets': 'landfill',
    'Plastic Utensil': 'landfill',
    'Plastic Wrap': 'landfill',
    'Straw': 'landfill'
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your model and number of classes from trash_model.py
from trash_model import TrashModel, num_classes

# Initialize and load the model weights (assumes weights saved in 'trash_model.pth')
model = TrashModel(num_classes)
model.load_state_dict(torch.load("trash_model.pth", map_location=device))
model.to(device)
model.eval()

# Define the image transformation (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # Preprocess the image
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()

    # Retrieve trash type and disposal bin from updated lists
    trash_type = trash_types[predicted_idx]
    disposal_bin = trash_to_bin.get(trash_type, "Unknown")

    # Return the results as JSON
    return jsonify({
        "trash_type": trash_type,
        "disposal_bin": disposal_bin
    })

if __name__ == "__main__":
    app.run(debug=True)
