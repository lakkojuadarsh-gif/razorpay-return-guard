import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("1. Loading dataset...")
# Load the dataset we generated earlier
df = pd.read_csv("razorpay_checkout_data.csv")

print("2. Preprocessing text data...")
# Convert text columns like 'payment_method' into numbers that the AI can understand
le = LabelEncoder()
df['payment_method'] = le.fit_transform(df['payment_method'])

# Save the encoder so we can use it in our live API later
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Separate features (X) from the answer target (y)
X = df.drop(columns=["transaction_id", "user_id", "is_returned"])
y = df["is_returned"]

print("3. Splitting data into Training and Test sets...")
# We use a 20% "held-out" test set to honestly evaluate performance, fulfilling the problem statement rules!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("4. Training the XGBoost AI model...")
# Initialize and train our model. We adjust scale_pos_weight because returns are less frequent than normal checkouts
model = XGBClassifier(scale_pos_weight=2, random_state=42)
model.fit(X_train, y_train)

print("5. Evaluating the model's accuracy...")
# Test the AI on data it has never seen before
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE REPORT ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed breakdown (Precision & Recall):")
print(classification_report(y_test, y_pred))

# Save the trained AI model to a file
with open("razorpay_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Success! Trained AI model saved as 'razorpay_model.pkl'")
