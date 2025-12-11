from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('app/model.joblib')
le = joblib.load('app/label_encoder.joblib')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins 
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
@app.get('/')
def home():
    return {"message": "Crop Prediction API - Send GET request to /predict"}

@app.get('/predict')
def predict(ph: float, n: float, p: float, k: float, 
            temp: float, humidity: float, rainfall: float):
    
    features = np.array([[ph, n, p, k, temp, humidity, rainfall]])
     
    prediction = model.predict(features)
    crop = le.inverse_transform(prediction)
    

    return {"predicted_crop": crop[0].item()}
