from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import sys
import os
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import DataPreprocessor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn for telecom/banking",
    version="1.0.0"
)

# Load model and preprocessor on startup
model = None
preprocessor = None

@app.on_event("startup")
async def load_model():
    global model, preprocessor
    try:
        model = joblib.load('models/best_churn_model.joblib')
        preprocessor = DataPreprocessor()
        preprocessor.load_artifacts('models/')
        print("Model and preprocessor loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# Input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    # Updated schema_extra for Pydantic v2 support
    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }

@app.get("/")
async def root():
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    """Predict churn probability for a single customer"""
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([customer.model_dump()]) # Using model_dump() for Pydantic V2
        
        # Preprocess
        X_processed, _ = preprocessor.preprocess(input_data, training=False)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Business interpretation
        risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability[1]),
            "retention_probability": float(probability[0]),
            "risk_level": risk_level,
            "recommendation": "Immediate retention offer" if risk_level == "High" else "Monitor closely" if risk_level == "Medium" else "Standard service"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(customers: List[CustomerData]):
    """Predict churn for multiple customers"""
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([c.model_dump() for c in customers])
        
        # Preprocess
        X_processed, _ = preprocessor.preprocess(input_data, training=False)
        
        # Predict
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            risk_level = "High" if prob[1] > 0.7 else "Medium" if prob[1] > 0.4 else "Low"
            results.append({
                "customer_id": i,
                "churn_prediction": int(pred),
                "churn_probability": float(prob[1]),
                "risk_level": risk_level
            })
        
        return {
            "total_customers": len(results),
            "predicted_churners": sum([r['churn_prediction'] for r in results]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
