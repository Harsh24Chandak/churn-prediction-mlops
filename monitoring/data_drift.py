import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from sklearn.model_selection import train_test_split
import os

os.makedirs('monitoring', exist_ok=True)

# Load reference and current data
reference_data = pd.read_csv('data/X_train.csv')
current_data = pd.read_csv('data/X_test.csv')

# Add target back for monitoring
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

reference_data['target'] = y_train
current_data['target'] = y_test

# Create report
report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset()
])

report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=ColumnMapping(target='target')
)

# Save HTML report
report.save_html('monitoring/drift_report.html')
print("Drift report saved to monitoring/drift_report.html")
