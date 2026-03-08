import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('../data', exist_ok=True)

# Load data
try:
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Dataset not found. Downloading...")
    import urllib.request
    url = 'https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    try:
        urllib.request.urlretrieve(url, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except Exception as e:
        print(f"Failed to download from first URL: {e}")
        # fallback to another commonly used repo
        url2 = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        urllib.request.urlretrieve(url2, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")

# Convert TotalCharges to numeric (it's object type)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Quick visualization
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.savefig('../data/churn_distribution.png')

# Correlation with tenure
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Tenure vs Churn')
plt.savefig('../data/tenure_churn.png')

print("EDA Complete! Check data/ folder for plots.")
