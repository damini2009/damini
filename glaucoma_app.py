import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# 🔍 Load your trained model (ResNet50)
def load_glaucoma_model(path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

# ✅ Load the model
model = load_glaucoma_model("models/glaucoma.pth")

# 🧠 Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# 🔁 Define predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        prediction = probs.tolist()

    return jsonify({'prediction': prediction})

# 🚀 Run the app on port 5002
if __name__ == '__main__':
    app.run(debug=True, port=5002)