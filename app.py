from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load your trained model
try:
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "🚗 Car Price Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    print("✅ Received request at /predict")
    
    try:
        data = request.get_json(force=True)
        print("📦 Received data:", data)

        # Extract and validate features
        year = int(data['year'])
        mileage = float(data['mileage'])
        engine_size = float(data['engine_size'])

        # Format the input as required by the model
        input_data = np.array([[year, mileage, engine_size]])
        prediction = model.predict(input_data)

        return jsonify({'predicted_price': float(prediction[0])})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
