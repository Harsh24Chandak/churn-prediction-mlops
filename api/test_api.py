import requests
import json

# Test single prediction
url = "http://localhost:8000/predict"

# High risk customer (month-to-month, low tenure, high charges)
customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.00,
    "TotalCharges": 190.00
}

response = requests.post(url, json=customer)
print("Single Prediction:")
try:
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}, Response: {response.text}")

# Test batch prediction
batch_url = "http://localhost:8000/predict_batch"
customers = [customer, customer]
response = requests.post(batch_url, json=customers)
print("\nBatch Prediction:")
try:
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}, Response: {response.text}")
