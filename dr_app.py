from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

# Load ONNX model
session = ort.InferenceSession("models/dr_model.onnx")
input_name = session.get_inputs()[0].name

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    try:
        # Open and preprocess image
        img = Image.open(file).convert('RGB')
        img = img.resize((456, 456))  # adjust to model input size
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

        # Inference
        output = session.run(None, {input_name: img_array})
        return jsonify({"prediction": output[0].tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Running on http://127.0.0.1:5000")
    app.run(debug=True)