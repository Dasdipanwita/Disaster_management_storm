import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import pickle

# Load the data
print("Loading storm data...")
df = pd.read_csv('storms.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Display basic info about the dataset
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

# Check for missing values in the original data
print("\nMissing values in original data:")
print(df[['lat', 'long', 'wind', 'pressure']].isnull().sum())

# Replace -999 values with NaN (common missing data indicator in weather data)
print("\nReplacing -999 values with NaN...")
df_clean = df.copy()

# Replace -999 with NaN in numeric columns
numeric_cols = ['wind', 'pressure', 'tropicalstorm_force_diameter', 'hurricane_force_diameter']
for col in numeric_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(-999, np.nan)
        df_clean[col] = df_clean[col].replace(-1998, np.nan)  # Another missing indicator

# Check missing values after replacement
print("\nMissing values after replacing -999/-1998 with NaN:")
print(df_clean[['lat', 'long', 'wind', 'pressure']].isnull().sum())

# Option 1: Remove rows with missing values in key features
print("\nRemoving rows with missing values in key features...")
df_clean = df_clean.dropna(subset=['lat', 'long', 'wind', 'pressure'])

print(f"Dataset shape after removing missing values: {df_clean.shape}")

# Prepare features and target
X = df_clean[['lat', 'long', 'wind', 'pressure']]
y = df_clean['status']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"No missing values in features: {X.isnull().sum().sum() == 0}")

# Check target variable distribution
print(f"\nTarget variable distribution:")
print(y.value_counts())

# Encode target variable if it's categorical
if y.dtype == 'object':
    print("\nEncoding categorical target variable...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Encoded classes: {le.classes_}")
else:
    y_encoded = y

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

# Verify no missing values in training data
print(f"Missing values in X_train before imputation: {X_train.isnull().sum().sum()}")
print(f"Missing values in X_test before imputation: {X_test.isnull().sum().sum()}")

# Impute missing values
print("\nImputing missing values using SimpleImputer...")
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

print(f"Missing values in X_train after imputation: {np.isnan(X_train).sum()}")
print(f"Missing values in X_test after imputation: {np.isnan(X_test).sum()}")

# Train the LogisticRegression model
print("\nTraining LogisticRegression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("Model training completed successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
if y.dtype == 'object':
    print(classification_report(y_test, y_pred, target_names=le.classes_))
else:
    print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix plot saved to confusion_matrix.png")

# Feature importance (coefficients for logistic regression)
print("\nFeature Coefficients:")
feature_names = ['lat', 'long', 'wind', 'pressure']
coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# Save the trained model and preprocessors
print("\nSaving model, imputer, and label encoder...")
with open('storm_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('storm_imputer.pkl', 'wb') as imputer_file:
    pickle.dump(imputer, imputer_file)
if 'le' in locals() and y.dtype == 'object':
    with open('storm_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(le, encoder_file)
print("Assets saved to storm_model.pkl, storm_imputer.pkl, and storm_encoder.pkl")

print("\nModel training and evaluation completed successfully!")
