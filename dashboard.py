import streamlit as st
import pandas as pd
import pickle

# Set page title and a clean Razorpay-inspired look
st.set_page_config(page_title="Razorpay ReturnGuard", layout="wide")

st.title("🛡️ Razorpay ReturnGuard™")
st.subheader("AI-Powered Risk Mitigation Suite for Smart Checkout Networks")
st.write("---")

# 1. Safely load our trained AI model and text encoder right inside Streamlit
@st.cache_resource
def load_ai_assets():
    try:
        with open("razorpay_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        return None, None

model, le = load_ai_assets()

# Create a clean layout with two parallel columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("🛒 Simulate Checkout Event")
    st.write("Adjust the buyer behavior sliders below to test how the AI evaluates risk in real-time.")
    
    payment_method = st.selectbox(
        "Payment Method Preferred by Buyer",
        ["UPI", "Credit_Card", "Debit_Card", "COD"],
        help="In India, Cash on Delivery (COD) carries significantly higher return and RTO rates."
    )
    
    size_bracketing = st.checkbox(
        "Detect Multi-Size Item Bracketing?",
        value=False,
        help="Check this if the buyer has added the exact same product model in multiple sizes."
    )
    
    historical_return_rate = st.slider(
        "Buyer's Cross-Merchant Historical Return Rate (%)",
        min_value=0, max_value=100, value=10,
    )
    
    high_risk_pincode = st.checkbox(
        "Is Shipping Destination a High-Risk Pincode?",
        value=False,
    )
    
    order_value = st.number_input(
        "Total Order Cart Value (INR)",
        min_value=500, max_value=50000, value=4500, step=500
    )

with col2:
    st.header("⚡ Live AI Risk Gateway Response")
    st.write("Click below to pass the transaction envelope payload into the built-in scoring pipeline.")
    
    if st.button("Analyze Transaction", type="primary"):
        if model is None or le is None:
            st.error("Error: Could not find 'razorpay_model.pkl' or 'label_encoder.pkl' in the workspace. Please make sure they exist in your project folder.")
        else:
            # 2. Re-create the prediction data mapping locally inside Streamlit
            input_data = {
                "size_bracketing_detected": [1 if size_bracketing else 0],
                "historical_return_rate": [float(historical_return_rate / 100.0)],
                "payment_method": [payment_method],
                "order_value_inr": [int(order_value)],
                "high_risk_pincode": [1 if high_risk_pincode else 0]
            }
            df_input = pd.DataFrame(input_data)
            
            # Map categories to numbers using the loaded encoder
            try:
                df_input['payment_method'] = le.transform(df_input['payment_method'])
            except:
                df_input['payment_method'] = 0

            # 🛠️ FIXED: Extract probability array correctly using .item() to avoid scalar errors
            probabilities = model.predict_proba(df_input)
            raw_probability_score = probabilities[0, 1]
            score = round(float(raw_probability_score) * 100, 2)
            
            # 3. Defensive Gate logic
            if score < 40.0:
                action = "ALLOW_NORMAL_CHECKOUT"
                details = "Safe customer behavior. All payment options unlocked."
                st.success(f"### RISK ASSESSMENT: SAFE ({score}%)")
                st.info(f"**Gate Automation Workflow Triggered:** \n\n `{action}` \n\n {details}")
            elif 40.0 <= score <= 75.0:
                action = "DISABLE_CASH_ON_DELIVERY"
                details = "Elevated return risk. Cash on Delivery disabled to protect merchant margins."
                st.warning(f"### RISK ASSESSMENT: ELEVATED RISK ({score}%)")
                st.error(f"**Gate Automation Workflow Triggered:** \n\n `{action}` \n\n {details}")
            else:
                action = "ENFORCE_FINAL_SALE_OTP"
                details = "High return risk detected. Enforce SMS OTP verification and mark cart items as non-returnable."
                st.error(f"### RISK ASSESSMENT: HIGH FRAUD RISK ({score}%)")
                st.error(f"**Gate Automation Workflow Triggered:** \n\n `{action}` \n\n {details}")
                
            # Financial metrics calculations
            st.write("---")
            st.markdown("### 📊 Business Financial Optimization Metrics")
            if action != "ALLOW_NORMAL_CHECKOUT":
                st.metric(
                    label="Estimated Return Logistics Cost Avoided", 
                    value=f"₹{round(order_value * 0.08)} INR",
                    delta="Loss Stopped"
                )
            else:
                st.metric(
                    label="Estimated Friction Revenue Maintained", 
                    value=f"₹{round(order_value * 0.02)} INR", 
                    delta="Conversion Protected", 
                    delta_color="inverse"
                )
