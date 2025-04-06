# test_request.py
import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "year": 2020,
    "mileage": 45000,
    "engine_size": 1.6
}

response = requests.post(url, json=data)
print("📦 Response:", response.json())
