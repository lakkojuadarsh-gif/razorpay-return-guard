import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

data = {
    "transaction_id": [f"pay_{100000 + i}" for i in range(n_samples)],
    "user_id": [f"cust_{np.random.randint(1000, 1500)}" for i in range(n_samples)],
    "size_bracketing_detected": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    "historical_return_rate": np.random.beta(a=1, b=5, size=n_samples),
    "payment_method": np.random.choice(["UPI", "Credit_Card", "Debit_Card", "COD"], size=n_samples, p=[0.4, 0.2, 0.1, 0.3]),
    "order_value_inr": np.random.randint(1500, 12000, size=n_samples),
    "high_risk_pincode": np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
}

df = pd.DataFrame(data)

def calculate_return_probability(row):
    prob = 0.05
    if row["size_bracketing_detected"] == 1: prob += 0.50
    if row["payment_method"] == "COD": prob += 0.25
    prob += row["historical_return_rate"] * 0.40
    if row["high_risk_pincode"] == 1: prob += 0.15
    return min(max(prob, 0.0), 0.95)

df["return_probability"] = df.apply(calculate_return_probability, axis=1)
df["is_returned"] = np.random.binomial(1, df["return_probability"])
df = df.drop(columns=["return_probability"])

df.to_csv("razorpay_checkout_data.csv", index=False)
print("Success! 'razorpay_checkout_data.csv' has been created.")
