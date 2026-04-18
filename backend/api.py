from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Climate-Aware Crop Yield Prediction API")

# Setup CORS for local dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_PATH = 'models/RandomForest_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
DATA_PATH = 'data/datasets/south_india_crop_data.csv'

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Calculate global Risk percentiles from historical data
    df_all = pd.read_csv(DATA_PATH)
    p25 = np.percentile(df_all['Yield'], 25)
    p75 = np.percentile(df_all['Yield'], 75)
else:
    model, preprocessor = None, None
    p25, p75 = 0, 0

class PredictionRequest(BaseModel):
    Crop: str
    Season: str
    State: str
    District: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    Rainfall_Required: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train models first.")
        
    try:
        # Construct DataFrame from request
        input_data = pd.DataFrame([{
            'Crop': request.Crop,
            'Season': request.Season,
            'State': request.State,
            'District': request.District,
            'Area': request.Area,
            'Annual_Rainfall': request.Annual_Rainfall,
            'Fertilizer': request.Fertilizer,
            'Pesticide': request.Pesticide,
            'N': request.N,
            'P': request.P,
            'K': request.K,
            'temperature': request.temperature,
            'humidity': request.humidity,
            'ph': request.ph,
            'Rainfall_Required': request.Rainfall_Required
        }])
        
        # Preprocess
        X_processed = preprocessor.transform(input_data)
        
        # Predict Yield
        yield_pred = model.predict(X_processed)[0]
        
        # Determine Risk
        if yield_pred < p25:
            risk_score = "High"
        elif yield_pred > p75:
            risk_score = "Low"
        else:
            risk_score = "Medium"
            
        return {
            "predicted_yield_tons_per_hectare": round(yield_pred, 3),
            "risk_score": risk_score,
            "message": "Prediction successful"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount visualizations folder to serve SHAP images
os.makedirs("visualizations", exist_ok=True)
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# To run: uvicorn backend.api:app --reload
