import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

class ChurnModelTrainer:
    def __init__(self, experiment_name="churn_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        self.best_model = None
        self.best_model_name = None
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train multiple models and log with MLflow"""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            with mlflow.start_run(run_name=name):
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Log parameters
                mlflow.log_param("model_type", name)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1: {f1:.4f}")
                print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Select best model based on F1 score (balance of precision and recall)
        self.best_model_name = max(results, key=lambda x: results[x]['f1'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best F1 Score: {results[self.best_model_name]['f1']:.4f}")
        
        return results
    
    def save_best_model(self, path='models/'):
        """Save the best model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.best_model, f'{path}best_churn_model.joblib')
        print(f"Best model saved to {path}best_churn_model.joblib")

if __name__ == "__main__":
    # Load processed data
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    
    # Train models
    trainer = ChurnModelTrainer()
    results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Save best model
    trainer.save_best_model()
    
    print("\nTraining complete! Check MLflow UI: mlflow ui")
