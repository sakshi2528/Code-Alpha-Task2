import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# ✅ Step 1: Sample training data
X = np.array([
    [2020, 45000, 1.6],
    [2018, 60000, 2.0],
    [2019, 50000, 1.8],
    [2017, 70000, 1.4],
    [2021, 30000, 2.0]
])

y = np.array([20000, 15000, 18000, 12000, 22000])

# ✅ Step 2: Train the model
model = LinearRegression()
model.fit(X, y)

# ✅ Step 3: Save the trained model
os.makedirs('model', exist_ok=True)
with open('model/car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
