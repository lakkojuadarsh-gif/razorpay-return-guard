# 🛡️ Razorpay ReturnGuard™
### AI-Powered Risk Mitigation Suite for Smart Checkout Networks

Razorpay ReturnGuard™ is a defensive risk management middleware engineered to integrate directly into the Razorpay Magic Checkout ecosystem. Built for the Razorpay Buildathon (Track 02: AI Risk Manager), this application tackles a massive multi-million dollar problem for Indian e-commerce merchants: high Return-to-Origin (RTO) rates and strategic return policy abuse.

---

## System Architecture
The core architecture consists of three interconnected layers working together in real time:
1. **The AI Brain (`train_model.py`)**: An optimized XGBoost Machine Learning pipeline trained on structured transaction data arrays, using class-weight scaling to balance heavily imbalanced checkout profiles.
2. **The Gateway Engine (`app.py`)**: A blazing-fast FastAPI backend serving a live prediction endpoint (`/v1/risk/score`) that calculates numeric risk probabilities within milliseconds.
3. **The Interactive Portal (`dashboard.py`)**: A Streamlit Merchant Dashboard visualizer that models real-world e-commerce checkout scenarios and automates defensive workflows.

---

## The 3 Risk Operational Scenarios

Based on the calculated risk severity, the gateway executes automated defensive workflows:

*   **SAFE PURCHASES (<40% risk)**: Triggers `ALLOW_NORMAL_CHECKOUT` to bypass security friction and ensure maximum payment success rates for trusted buyers.
*   **ELEVATED RISKS (40%-75% risk)**: Triggers `DISABLE_CASH_ON_DELIVERY` to automatically enforce prepaid UPI/card conversion when multi-size item bracketing is detected, slashing impulsive doorstep rejections.
*   **HIGH FRAUD THREATS (>75% risk)**: Triggers `ENFORCE_FINAL_SALE_OTP`, injecting multi-factor validation and marking items as non-refundable to lock out serial return abusers.

---

## Meeting 'The Bar' (Honest Metrics)
- **True Evaluation**: The model explicitly isolates its training metrics by validating on a 20% completely unseen, held-out test dataset to report true Precision and Recall.
- **False-Positive Optimization**: The system features an economic loss calculator that addresses false-positive costs by mathematically evaluating user conversion friction against saved reverse logistics shipping overheads.

---

## Local Installation & Setup

1. Clone this repository and open it in VS Code.
2. Install the requirements in your terminal:
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi uvicorn streamlit
   ```
3. Generate the mockup dataset:
   ```bash
   python generate_data.py
   ```
4. Train the XGBoost AI model:
   ```bash
   python train_model.py
   ```
5. Start the FastAPI backend engine:
   ```bash
   python -m uvicorn app:app --reload
   ```
6. Launch the interactive Streamlit dashboard in a separate terminal:
   ```bash
   python -m streamlit run dashboard.py
   ```
