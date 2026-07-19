from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List

# Load model artifacts
model = joblib.load('models/best_churn_model.joblib')
encoders = joblib.load('models/encoders.joblib')
scaler = joblib.load('models/scaler.joblib')

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn for telecom/banking",
    version="1.0.0"
)

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

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs", "health": "/health"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    try:
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Encode categoricals
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))
        
        # Scale numericals
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
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
    try:
        input_data = pd.DataFrame([c.model_dump() for c in customers])
        
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
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
