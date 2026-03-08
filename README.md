# Customer Churn Prediction with MLOps

## 🚀 Quick Start
```bash
git clone https://github.com/yourusername/churn-prediction-mlops.git
cd churn-prediction-mlops
pip install -r requirements.txt
python src/preprocessing.py
python src/model_trainer.py
uvicorn api.main:app --reload
```

## 🛠️ Tech Stack
- ML: Scikit-learn, XGBoost, Pandas
- API: FastAPI, Uvicorn
- MLOps: MLflow (experiment tracking), Evidently (drift detection), Docker
- Deployment: Render (cloud)

## 📈 Results
- Best Model: XGBoost
- F1 Score: 0.82
- Precision: 0.81 (few false alarms)
- Recall: 0.83 (catch most churners)
- Business Impact: Can identify 83% of at-risk customers 30 days in advance

## 🐳 Docker
```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## 📡 API Endpoints
- `POST /predict` - Single customer prediction
- `POST /predict_batch` - Batch predictions
- `GET /health` - Health check

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 📊 MLflow Tracking
```bash
mlflow ui
# Open http://localhost:5000
```

## 🏗️ Architecture
`[Data] → [Preprocessing] → [Training (MLflow)] → [Model Registry] → [FastAPI] → [Docker] → [Render]`

## 👨‍💻 Author
Harsh Chandak
