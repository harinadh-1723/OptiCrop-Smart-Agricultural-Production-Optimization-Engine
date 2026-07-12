import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
csv_path = "Crop_recommendation.csv"

if not os.path.exists(csv_path):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete.")
else:
    print("Dataset already exists locally.")

# Step 2: Load the dataset
df = pd.read_csv(csv_path)
print("Initial Dataset Shape:", df.shape)

# Step 3: Rename columns to match the documentation
# N -> nitrogen, P -> phosphorous, K -> potassium
rename_dict = {
    'N': 'nitrogen',
    'P': 'phosphorous',
    'K': 'potassium'
}
# Only rename if columns exist
df.rename(columns=lambda x: rename_dict.get(x, x), inplace=True)
print("Columns after renaming:", list(df.columns))

# Step 4: Check for null values
print("Null values check:")
print(df.isnull().sum())

# Step 5: Handling Outliers
# The documentation specifies removing outliers in 'phosphorous' using IQR
Q1 = df['phosphorous'].quantile(0.25)
Q3 = df['phosphorous'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Phosphorous IQR: {IQR}, Bounds: [{lower_bound}, {upper_bound}]")
filter_mask = (df['phosphorous'] >= lower_bound) & (df['phosphorous'] <= upper_bound)
df = df.loc[filter_mask]
print("Dataset Shape after outlier removal:", df.shape)

# Step 6: Splitting features and label
y = df['label']
X = df.drop(['label'], axis=1)

print("Features Shape (X):", X.shape)
print("Target Shape (y):", y.shape)

# Step 7: Train-test split
# Using test_size=0.2 and random_state=0 as seen in the screenshots
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Step 8: Model Training (Logistic Regression)
# Note: Increased max_iter to 2000 to prevent ConvergenceWarning
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000, random_state=0)
model.fit(X_train, y_train)

# Step 9: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Serialize the model
model_path = "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model serialized and saved to '{model_path}'.")
