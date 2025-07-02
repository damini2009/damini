from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import os
import logging
from datetime import datetime

# HTML Template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical AI - Fundus Image Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            padding: 40px;
        }
        
        .upload-section {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .upload-area {
            border: 3px dashed #3498db;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        
        .upload-area:hover {
            border-color: #2980b9;
            background: #e3f2fd;
        }
        
        .upload-area.dragover {
            border-color: #27ae60;
            background: #e8f5e8;
        }
        
        .upload-icon {
            font-size: 3em;
            color: #3498db;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            
        }
        
        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .model-selection {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .model-card {
            background: white;
            border: 2px solid #ecf0f1;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 150px;
        }
        
        .model-card:hover {
            border-color: #3498db;
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .model-card.selected {
            border-color: #27ae60;
            background: #e8f5e8;
        }
        
        .model-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .results-section {
            margin-top: 40px;
        }
        
        .result-card {
            background: white;
            border: 1px solid #ecf0f1;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }
        
        .result-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .confidence-bar {
            background: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
            transition: width 0.5s ease;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fee;
            border: 1px solid #fcc;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .success {
            background: #efe;
            border: 1px solid #cfc;
            color: #363;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .image-preview {
            max-width: 300px;
            max-height: 300px;
            border-radius: 10px;
            margin: 20px auto;
            display: block;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 15px;
            }
            
            .main-content {
                padding: 20px;
            }
            
            .model-selection {
                flex-direction: column;
                align-items: center;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical AI System</h1>
            <p>Advanced Fundus Image Analysis for Diabetic Retinopathy, Glaucoma, and Retinal Disease Detection</p>
        </div>
        
        <div class="main-content">
            <div class="upload-section">
                <h2>📸 Upload Fundus Image</h2>
                <div class="upload-area" id="uploadArea">
                    <div class="upload-icon">📁</div>
                    <p>Drag and drop your fundus image here or click to browse</p>
                    <input type="file" id="fileInput" class="file-input" accept="image/*">
                    <button class="btn" onclick="document.getElementById('fileInput').click()">Choose File</button>
                </div>
                
                <div class="model-selection">
                    <div class="model-card" data-model="all">
                        <div class="model-icon">🔍</div>
                        <h3>All Models</h3>
                        <p>Complete Analysis</p>
                    </div>
                    <div class="model-card" data-model="dr">
                        <div class="model-icon">👁️</div>
                        <h3>Diabetic Retinopathy</h3>
                        <p>DR Detection</p>
                    </div>
                    <div class="model-card" data-model="glaucoma">
                        <div class="model-icon">🔍</div>
                        <h3>Glaucoma</h3>
                        <p>Glaucoma Detection</p>
                    </div>
                    <div class="model-card" data-model="retina">
                        <div class="model-icon">👁️‍🗨️</div>
                        <h3>Retinal Disease</h3>
                        <p>Retinal Analysis</p>
                    </div>
                </div>
                
                <button class="btn" id="analyzeBtn" disabled>🔬 Analyze Image</button>
            </div>
            
            <div id="imagePreview"></div>
            <div id="loading" class="loading" style="display: none;">
                <div class="spinner"></div>
                <p>Analyzing image with AI models...</p>
            </div>
            <div id="results"></div>
        </div>
    </div>

    <script>
        let selectedModel = 'all';
        let selectedFile = null;
        
        // Model selection
        document.querySelectorAll('.model-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                selectedModel = card.dataset.model;
            });
        });
        
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const imagePreview = document.getElementById('imagePreview');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }
            
            selectedFile = file;
            analyzeBtn.disabled = false;
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.innerHTML = `<img src="${e.target.result}" class="image-preview" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }
        
        // Analyze button
        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            loading.style.display = 'block';
            results.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const endpoint = selectedModel === 'all' ? '/predict/all' : `/predict/${selectedModel}`;
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                } else {
                    displayError(data.error);
                }
            } catch (error) {
                displayError('Network error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        });
        
        function displayResults(data) {
            let html = '<div class="success">✅ Analysis completed successfully!</div>';
            
            if (data.results) {
                // Multiple models
                Object.entries(data.results).forEach(([model, result]) => {
                    html += createResultCard(model, result);
                });
            } else {
                // Single model
                html += createResultCard(data.model, data.result);
            }
            
            results.innerHTML = html;
        }
        
        function createResultCard(model, result) {
            const modelNames = {
                'dr': 'Diabetic Retinopathy',
                'glaucoma': 'Glaucoma',
                'retina': 'Retinal Disease'
            };
            
            const confidencePercent = Math.round(result.confidence * 100);
            const confidenceColor = confidencePercent > 80 ? '#27ae60' : 
                                  confidencePercent > 60 ? '#f39c12' : '#e74c3c';
            
            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">${modelNames[model] || model}</div>
                        <div style="color: ${confidenceColor}; font-weight: bold;">
                            ${confidencePercent}% Confidence
                        </div>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <strong>Prediction:</strong> ${result.prediction}
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%; background: ${confidenceColor};"></div>
                    </div>
                    ${result.probabilities ? `
                        <div style="margin-top: 15px;">
                            <strong>Probabilities:</strong>
                            <ul style="margin-top: 5px; padding-left: 20px;">
                                ${result.probabilities.map((prob, idx) => 
                                    `<li>${(prob * 100).toFixed(1)}%</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        function displayError(message) {
            results.innerHTML = `<div class="error">❌ Error: ${message}</div>`;
        }
        
        // Auto-select "All Models" by default
        document.querySelector('[data-model="all"]').classList.add('selected');
    </script>
</body>
</html>
'''

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for models
dr_session = None
glaucoma_model = None
retina_model = None

# Model configurations
MODEL_CONFIGS = {
    'dr': {
        'model_path': 'models/dr_model.onnx',
        'input_size': (456, 456),  # Original ONNX model size
        'labels': ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
    },
    'glaucoma': {
        'model_path': 'models/glaucoma.pth',
        'input_size': (224, 224),  # Original PyTorch model size
        'labels': ['No Glaucoma', 'Glaucoma']
    },
    'retina': {
        'model_path': 'models/retina.pth',
        'input_size': (456, 456),  # Original PyTorch model size
        'labels': ['BRVO', 'CRVO', 'Normal', 'RAO']
    }
}

def load_models():
    """Load all models on startup"""
    global dr_session, glaucoma_model, retina_model
    
    try:
        # Load DR model (ONNX)
        logger.info("Loading DR model...")
        dr_session = ort.InferenceSession(MODEL_CONFIGS['dr']['model_path'])
        
        # Load Glaucoma model (PyTorch)
        logger.info("Loading Glaucoma model...")
        glaucoma_model = models.resnet50(pretrained=False)
        num_ftrs = glaucoma_model.fc.in_features
        glaucoma_model.fc = nn.Linear(num_ftrs, 2)
        glaucoma_model.load_state_dict(torch.load(MODEL_CONFIGS['glaucoma']['model_path'], map_location='cpu'))
        glaucoma_model.eval()
        
        # Load Retina model (PyTorch)
        logger.info("Loading Retina model...")
        retina_model = efficientnet_b0(pretrained=False)
        num_classes = len(MODEL_CONFIGS['retina']['labels'])
        retina_model.classifier[1] = torch.nn.Linear(1280, num_classes)
        retina_model.load_state_dict(torch.load(MODEL_CONFIGS['retina']['model_path'], map_location='cpu'))
        retina_model.eval()
        
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def preprocess_image(image, model_type):
    """Standardized preprocessing for all fundus images"""
    try:
        # Step 1: Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 2: Get the required input size for this specific model
        required_size = MODEL_CONFIGS[model_type]['input_size']
        image = image.resize(required_size, Image.Resampling.LANCZOS)
        
        # Step 3: Convert to numpy array and normalize pixel values to 0-1
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Step 4: Model-specific preprocessing
        if model_type == 'dr':
            # For ONNX model - keep as numpy array with shape (1, H, W, 3)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        else:
            # For PyTorch models - convert to tensor and apply ImageNet normalization
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC to CHW
            
            # Apply ImageNet normalization (mean and std)
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            img_tensor = (img_tensor - imagenet_mean) / imagenet_std
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            return img_tensor
            
    except Exception as e:
        logger.error(f"Error in standardized preprocessing: {str(e)}")
        raise

def predict_dr(image):
    """Predict diabetic retinopathy"""
    try:
        if dr_session is None:
            raise ValueError("DR model not loaded")
            
        input_name = dr_session.get_inputs()[0].name
        preprocessed_img = preprocess_image(image, 'dr')
        output = dr_session.run(None, {input_name: preprocessed_img})
        probabilities = np.array(output[0][0])  # type: ignore # Convert to numpy array
        prediction_idx = int(np.argmax(probabilities))
        
        return {
            'prediction': MODEL_CONFIGS['dr']['labels'][prediction_idx],
            'confidence': float(probabilities[prediction_idx]),
            'probabilities': probabilities.tolist()
        }
    except Exception as e:
        logger.error(f"DR prediction error: {str(e)}")
        raise

def predict_glaucoma(image):
    """Predict glaucoma"""
    try:
        if glaucoma_model is None:
            raise ValueError("Glaucoma model not loaded")
            
        preprocessed_img = preprocess_image(image, 'glaucoma')
        
        with torch.no_grad():
            outputs = glaucoma_model(preprocessed_img)
            probs = torch.softmax(outputs, dim=1)
            prediction_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0][prediction_idx].item())
        
        return {
            'prediction': MODEL_CONFIGS['glaucoma']['labels'][prediction_idx],
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }
    except Exception as e:
        logger.error(f"Glaucoma prediction error: {str(e)}")
        raise

def predict_retina(image):
    """Predict retinal diseases"""
    try:
        if retina_model is None:
            raise ValueError("Retina model not loaded")
            
        preprocessed_img = preprocess_image(image, 'retina')
        
        with torch.no_grad():
            output = retina_model(preprocessed_img)
            probs = torch.softmax(output, dim=1)
            prediction_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0][prediction_idx].item())
        
        return {
            'prediction': MODEL_CONFIGS['retina']['labels'][prediction_idx],
            'confidence': confidence,
            'probabilities': probs[0].tolist()
        }
    except Exception as e:
        logger.error(f"Retina prediction error: {str(e)}")
        raise

@app.route('/')
def home():
    """Serve the home page with navigation"""
    return render_template('home.html')

@app.route('/flow1')
def flow1():
    """Serve the Flow 1 page with the flowchart image"""
    return render_template('flow1.html')

@app.route('/flow2', methods=['GET', 'POST'])
def flow2():
    """Serve the Flow 2 questionnaire and process answers"""
    if request.method == 'GET':
        return render_template('flow2.html')
    else:
        data = request.get_json()
        # Map answers to models as per the flowchart
        models = set()
        if data.get('q1') == 'yes':
            models.add('Diabetic Retinopathy (DR)')
        if data.get('q2') == 'yes':
            models.add('Retinal Vascular Occlusion (RVO)')
            models.add('Glaucoma')
        if data.get('q3') == 'yes':
            models.add('RVO')
            models.add('Glaucoma')
        if data.get('q4') == 'yes':
            models.add('DR')
            models.add('RVO')
            models.add('Glaucoma')
        if data.get('q5') == 'yes':
            models.add('Glaucoma')
        if data.get('q6') == 'yes':
            models.add('Glaucoma')
        # Clean up model names for display
        display_models = []
        for m in models:
            if m == 'RVO':
                display_models.append('Retinal Vascular Occlusion (RVO)')
            elif m == 'DR':
                display_models.append('Diabetic Retinopathy (DR)')
            else:
                display_models.append(m)
        return jsonify({'models': sorted(set(display_models))})

@app.route('/flowchart.jpg')
def flowchart_img():
    return send_from_directory('.', 'flowchart.jpg')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = all([dr_session, glaucoma_model, retina_model])
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/<model_type>', methods=['POST'])
def predict_single(model_type):
    """Predict using a specific model"""
    if model_type not in ['dr', 'glaucoma', 'retina']:
        return jsonify({"error": "Invalid model type"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        
        # Validate image
        if image.size[0] < 100 or image.size[1] < 100:
            return jsonify({"error": "Image too small"}), 400
        
        # Make prediction based on model type
        if model_type == 'dr':
            result = predict_dr(image)
        elif model_type == 'glaucoma':
            result = predict_glaucoma(image)
        else:  # retina
            result = predict_retina(image)
        
        return jsonify({
            'model': model_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """Predict using all models"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        
        # Validate image
        if image.size[0] < 100 or image.size[1] < 100:
            return jsonify({"error": "Image too small"}), 400
        
        results = {}
        
        # Predict with all models
        results['dr'] = predict_dr(image)
        results['glaucoma'] = predict_glaucoma(image)
        results['retina'] = predict_retina(image)
        
        return jsonify({
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multi-prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/preprocessing-info')
def preprocessing_info():
    """Information about the standardized preprocessing pipeline"""
    return jsonify({
        'standardized_preprocessing': {
            'color_format': 'RGB',
            'normalization': 'Pixel values scaled to 0-1 range',
            'imagenet_normalization': 'Applied for PyTorch models (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
            'resampling_method': 'LANCZOS for high-quality resizing',
            'steps': [
                '1. Convert to RGB format',
                '2. Resize to model-specific input size',
                '3. Normalize pixel values to 0-1 range',
                '4. Apply model-specific preprocessing (ONNX vs PyTorch)'
            ]
        },
        'model_input_sizes': {
            'dr': '456x456 pixels (ONNX model)',
            'glaucoma': '224x224 pixels (PyTorch ResNet50)',
            'retina': '456x456 pixels (PyTorch EfficientNet)'
        },
        'model_compatibility': {
            'dr': 'ONNX model - numpy array format',
            'glaucoma': 'PyTorch model - tensor with ImageNet normalization',
            'retina': 'PyTorch model - tensor with ImageNet normalization'
        },
        'model_purposes': {
            'dr': 'Diabetic Retinopathy Detection',
            'glaucoma': 'Glaucoma Detection',
            'retina': 'Retinal Vein Occlusion Detection (BRVO, CRVO, Normal, RAO)'
        }
    })

@app.route('/run_selected_models', methods=['POST'])
def run_selected_models():
    if 'file' not in request.files or 'models' not in request.form:
        return jsonify({'error': 'Missing file or models'}), 400
    file = request.files['file']
    models_str = request.form['models']
    models_to_run = [m.strip().lower() for m in models_str.split(',')]
    image = Image.open(file.stream).convert('RGB')
    results = {}
    for model in models_to_run:
        if 'diabetic' in model or model == 'dr':
            results['Diabetic Retinopathy (DR)'] = predict_dr(image)
        elif 'glaucoma' in model:
            results['Glaucoma'] = predict_glaucoma(image)
        elif 'retinal' in model or 'rvo' in model:
            results['Retinal Vascular Occlusion (RVO)'] = predict_retina(image)
    return jsonify({'results': results})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("🚀 EyeRa starting on http://127.0.0.1:5000")
        print("📊 Available endpoints:")
        print("   - GET  /health - Health check")
        print("   - GET  /preprocessing-info - Standardized preprocessing details")
        print("   - POST /predict/dr - Diabetic Retinopathy detection")
        print("   - POST /predict/glaucoma - Glaucoma detection")
        print("   - POST /predict/retina - Retinal Vein Occlusion detection")
        print("   - POST /predict/all - All models prediction")
        print("   - GET  / - Web interface")
        print("\n🔬 Standardized Preprocessing:")
        print("   - DR model: 456x456 pixels (ONNX)")
        print("   - Glaucoma model: 224x224 pixels (PyTorch)")
        print("   - Retinal Vein Occlusion model: 456x456 pixels (PyTorch)")
        print("   - RGB format with 0-1 normalization")
        print("   - ImageNet normalization for PyTorch models")
        print("\n👁️ EyeRa - Medical Conditions Detected:")
        print("   - Diabetic Retinopathy: No DR, Mild, Moderate, Severe, Proliferative")
        print("   - Glaucoma: No Glaucoma, Glaucoma")
        print("   - Retinal Vein Occlusion: BRVO, CRVO, Normal, RAO")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check model files.") 