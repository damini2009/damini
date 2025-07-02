import requests
import time
import json

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get('http://127.0.0.1:5000/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return data['models_loaded']
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_model(model_type, image_path):
    """Test a single model prediction"""
    print(f"🔬 Testing {model_type} model...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'http://127.0.0.1:5000/predict/{model_type}', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {model_type} prediction successful:")
            print(f"   Prediction: {data['result']['prediction']}")
            print(f"   Confidence: {data['result']['confidence']:.2%}")
            return True
        else:
            print(f"❌ {model_type} prediction failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ {model_type} prediction error: {e}")
        return False

def test_all_models(image_path):
    """Test all models prediction"""
    print("🔬 Testing all models...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://127.0.0.1:5000/predict/all', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ All models prediction successful:")
            for model, result in data['results'].items():
                print(f"   {model}: {result['prediction']} ({result['confidence']:.2%})")
            return True
        else:
            print(f"❌ All models prediction failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ All models prediction error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Medical AI System Test Suite")
    print("=" * 50)
    
    # Test image path - update this to your actual image path
    test_image = 'img2.jpg'  # or 'hr.jpg'
    
    # Check if test image exists
    try:
        with open(test_image, 'rb') as f:
            print(f"✅ Test image found: {test_image}")
    except FileNotFoundError:
        print(f"❌ Test image not found: {test_image}")
        print("Please update the test_image variable to point to an existing image file.")
        return
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Test health check
    models_loaded = test_health_check()
    if not models_loaded:
        print("❌ Models not loaded. Please check if the server is running and models are available.")
        return
    
    print("\n" + "=" * 50)
    
    # Test individual models
    models = ['dr', 'glaucoma', 'retina']
    for model in models:
        test_single_model(model, test_image)
        print()
    
    print("=" * 50)
    
    # Test all models
    test_all_models(test_image)
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    print("🌐 You can now access the web interface at: http://127.0.0.1:5000")

if __name__ == '__main__':
    main() 