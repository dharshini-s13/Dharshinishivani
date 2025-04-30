# 📌 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# 📌 2. Load the dataset
df = pd.read_csv("Churn_Modelling.csv")
df.head()
python
Copy
Edit
# 📌 3. Data Cleaning
# Drop irrelevant columns that don't help with prediction
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())
python
Copy
Edit
# 📌 4. Exploratory Data Analysis (EDA)

# Basic summary statistics
print(df.describe())

# Churn Distribution
sns.countplot(x='Exited', data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
python
Copy
Edit
# 📌 5. Feature Engineering

# Convert categorical columns using one-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
python
Copy
Edit
# 📌 6. Model Development - Logistic Regression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("🔍 Logistic Regression Evaluation")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
python
Copy
Edit
# 📌 7. Model Development - Random Forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("🔍 Random Forest Evaluation")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
python
Copy
Edit
# 📌 8. Classification Accuracy Comparison

log_accuracy = accuracy_score(y_test, y_pred_log)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"✅ Logistic Regression Accuracy: {log_accuracy:.4f}")
print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")









