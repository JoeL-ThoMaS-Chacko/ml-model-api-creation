from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load('app/model.joblib')
le = joblib.load('app/label_encoder.joblib')
app = FastAPI()

@app.get('/')
def home():
    return {"message": "Crop Prediction API - Send GET request to /predict"}

@app.get('/predict')
def predict(ph: float, n: float, p: float, k: float, 
            temp: float, humidity: float, rainfall: float):
    
    # Create feature array
    features = np.array([[ph, n, p, k, temp, humidity, rainfall]])
     
    # Make prediction
    prediction = model.predict(features)
    crop = le.inverse_transform(prediction)
    
    return {"predicted_crop": crop[0].item()}