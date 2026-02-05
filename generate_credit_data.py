import pandas as pd
import numpy as np
import random

# CONFIGURATION
NUM_APPLICANTS = 20000
DEFAULT_RATE = 0.05  # 5% of people default (Real-world imbalance)

# HELPER: Generate Data
def generate_data():
    data = []
    
    for i in range(NUM_APPLICANTS):
        # 1. Demographics
        id = f"APP-{100000+i}"
        age = int(np.random.normal(32, 8)) # Avg age 32
        age = max(21, min(60, age)) # Clip between 21 and 60
        
        city_tier = np.random.choice(['Tier_1', 'Tier_2', 'Tier_3'], p=[0.5, 0.3, 0.2])
        
        # 2. Financials
        # Income varies by City Tier
        base_income = 800000 if city_tier == 'Tier_1' else 500000
        annual_income = int(max(300000, np.random.normal(base_income, 200000)))
        
        # CIBIL Score (Skewed towards good scores, but some bad)
        cibil_score = int(min(900, max(300, np.random.normal(750, 80))))
        
        # UPI Activity (Higher income usually = Higher UPI volume)
        upi_txns = int(annual_income / 15000) + np.random.randint(-10, 20)
        upi_txns = max(0, upi_txns)
        
        # 3. Loan Details
        loan_amount = int(np.random.choice([50000, 100000, 200000, 500000], p=[0.4, 0.3, 0.2, 0.1]))
        
        # 4. Target Variable: DEFAULT (Risk Logic)
        # Probability of default increases if:
        # - Low CIBIL
        # - High Loan to Income Ratio
        # - Low UPI usage
        
        prob_default = DEFAULT_RATE
        
        if cibil_score < 650: prob_default += 0.25
        if (loan_amount / annual_income) > 0.4: prob_default += 0.15
        if upi_txns < 5: prob_default += 0.10
        if city_tier == 'Tier_3': prob_default += 0.05
        
        # Cap probability at 1.0
        prob_default = min(0.9, prob_default)
        
        # Assign Default Status (1 = Default, 0 = Paid)
        is_default = np.random.choice([1, 0], p=[prob_default, 1-prob_default])
        
        data.append({
            'Applicant_ID': id,
            'Age': age,
            'City_Tier': city_tier,
            'Annual_Income': annual_income,
            'CIBIL_Score': cibil_score,
            'UPI_Txns_Monthly': upi_txns,
            'Loan_Amount': loan_amount,
            'Loan_Tenure_Months': np.random.choice([12, 24, 36]),
            'Default_Status': is_default
        })
        
    return pd.DataFrame(data)

# GENERATE & SAVE
df = generate_data()

# Validate Imbalance
default_count = df['Default_Status'].sum()
print(f"Dataset Generated: {NUM_APPLICANTS} Applicants.")
print(f"Defaulters: {default_count} ({default_count/NUM_APPLICANTS*100:.2f}%)")
print("Data saved to 'credit_risk_dataset.csv'")

df.to_csv('credit_risk_dataset.csv', index=False)