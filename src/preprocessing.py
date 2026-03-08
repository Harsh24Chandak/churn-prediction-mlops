import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load and clean raw data"""
        df = pd.read_csv(filepath)
        
        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        # Drop customerID (not useful for prediction)
        df.drop('customerID', axis=1, inplace=True)
        
        return df
    
    def preprocess(self, df, training=True):
        """Main preprocessing pipeline"""
        df_processed = df.copy()
        
        # Separate target
        if 'Churn' in df_processed.columns:
            y = df_processed['Churn'].map({'Yes': 1, 'No': 0})
            df_processed.drop('Churn', axis=1, inplace=True)
        else:
            y = None
            
        # Identify categorical and numerical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        for col in categorical_cols:
            if training:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                # Handle unknown labels in inference using try-except or just transforming
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Scale numerical features
        if training:
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            self.feature_columns = df_processed.columns.tolist()
        else:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed, y
    
    def save_artifacts(self, path='models/'):
        """Save preprocessing artifacts"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.scaler, f'{path}scaler.joblib')
        joblib.dump(self.label_encoders, f'{path}label_encoders.joblib')
        joblib.dump(self.feature_columns, f'{path}feature_columns.joblib')
    
    def load_artifacts(self, path='models/'):
        """Load preprocessing artifacts"""
        self.scaler = joblib.load(f'{path}scaler.joblib')
        self.label_encoders = joblib.load(f'{path}label_encoders.joblib')
        self.feature_columns = joblib.load(f'{path}feature_columns.joblib')

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor()
    os.makedirs('data', exist_ok=True)
    df = preprocessor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X, y = preprocessor.preprocess(df, training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save artifacts
    preprocessor.save_artifacts()
    
    # Save processed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print(f"Preprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    print(f"Churn rate in test: {y_test.mean():.2%}")
