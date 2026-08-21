from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# 1. Initialize our FastAPI Gateway app
app = FastAPI(title="Razorpay ReturnGuard AI Gateway")

# 2. Load our trained AI model and text encoder
with open("razorpay_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 3. Define the exact format of the incoming checkout data
class CheckoutData(BaseModel):
    size_bracketing_detected: int  # 1 for Yes, 0 for No
    historical_return_rate: float # 0.0 to 1.0
    payment_method: str           # "UPI", "Credit_Card", "Debit_Card", or "COD"
    order_value_inr: int          # e.g., 4500
    high_risk_pincode: int        # 1 for Yes, 0 for No

# 4. Create the prediction endpoint
@app.post("/v1/risk/score")
def evaluate_checkout_risk(data: CheckoutData):
    # Prepare the raw data into a readable format for our AI model
    input_data = {
        "size_bracketing_detected": [data.size_bracketing_detected],
        "historical_return_rate": [data.historical_return_rate],
        "payment_method": [data.payment_method],
        "order_value_inr": [data.order_value_inr],
        "high_risk_pincode": [data.high_risk_pincode]
    }
    df_input = pd.DataFrame(input_data)
    
    # Transform the text payment method into a number using our encoder
    try:
        df_input['payment_method'] = le.transform(df_input['payment_method'])
    except ValueError:
        # If an unknown payment method is sent, default to standard number coding
        df_input['payment_method'] = 0

    # Calculate the exact probability percentage of a return happening
    probabilities = model.predict_proba(df_input)
    return_probability = float(probabilities[0][1]) * 100
    
    # 5. Define Razorpay Defensive Business Logic based on the AI score
    if return_probability < 40.0:
        action = "ALLOW_NORMAL_CHECKOUT"
        message = "Safe customer behavior. All payment options unlocked."
    elif 40.0 <= return_probability <= 75.0:
        action = "DISABLE_CASH_ON_DELIVERY"
        message = "Elevated return risk. Cash on Delivery disabled to protect merchant margins."
    else:
        action = "ENFORCE_FINAL_SALE_OTP"
        message = "High return risk detected. Enforce SMS OTP verification and mark cart items as non-returnable."

    # Return the clean evaluation response package back to the checkout site
    return {
        "risk_score_percentage": round(return_probability, 2),
        "recommended_action": action,
        "action_details": message
    }
