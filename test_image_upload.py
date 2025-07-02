import requests

# 🖼 Path to your test image - Fixed path syntax
file_path = r'X:\Fundus\Fundus\CRVO.jpg'  # Using raw string to avoid escape sequence issues

# ✅ Send POST request to your unified Flask backend
response = requests.post(
    'http://127.0.0.1:5000/predict/retina',  # Updated to use unified endpoint
    files={'file': open(file_path, 'rb')}
)

# ✅ Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())

# Alternative: Test all models
print("\n" + "="*50)
print("Testing all models...")
response_all = requests.post(
    'http://127.0.0.1:5000/predict/all',
    files={'file': open(file_path, 'rb')}
)

print("All Models Status Code:", response_all.status_code)
print("All Models Response:", response_all.json())