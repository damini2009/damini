from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision.models import efficientnet_b0
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)
CORS(app)

# 🧠 Labels (Change if different)
labels = ['BRVO', 'CRVO', 'Normal', 'RAO']

# 🧠 Load Model
def load_retina_model(path):
    model = efficientnet_b0(pretrained=False)
    num_classes = len(labels)  # Dynamic based on number of labels
    model.classifier[1] = torch.nn.Linear(1280, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

model = load_retina_model("models/retina.pth")

# 🖼 Preprocessing
transform = transforms.Compose([
    transforms.Resize((456, 456)),
    transforms.ToTensor()
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image = Image.open(request.files['file'].stream).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # type: ignore # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        pred_index = int(torch.argmax(output, dim=1))

    return jsonify({'prediction': labels[pred_index]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # 👈 Use different port (5001)