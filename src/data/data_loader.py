import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_RAW

def load_telecom_data():
    """Load Telecom Customer Churn Dataset"""
    print("Loading Telecom Customer Churn Dataset...")
    
    try:
        data = pd.read_csv(os.path.join(DATA_RAW, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        print(f"[OK] Loaded {len(data)} customers")
    except:
        print("\n[WARNING] Dataset not found!")
        print("\nDownload Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("2. Download: WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print(f"3. Place in: {DATA_RAW}")
        print("4. Run: python src/data/data_loader.py\n")
        sys.exit(1)
    
    return data

def transform_to_standard_format(data):
    """Transform telecom data to standard format"""
    print("Transforming data...")
    
    customers = pd.DataFrame({
        'customer_id': data['customerID'],
        'gender': data['gender'],
        'senior_citizen': data['SeniorCitizen'],
        'partner': data['Partner'],
        'dependents': data['Dependents'],
        'tenure': data['tenure'],
        'phone_service': data['PhoneService'],
        'internet_service': data['InternetService'],
        'contract': data['Contract'],
        'payment_method': data['PaymentMethod'],
        'monthly_charges': pd.to_numeric(data['MonthlyCharges'], errors='coerce'),
        'total_charges': pd.to_numeric(data['TotalCharges'], errors='coerce'),
        'churn': data['Churn'].map({'Yes': 1, 'No': 0})
    })
    
    customers['total_charges'].fillna(customers['monthly_charges'] * customers['tenure'], inplace=True)
    customers.dropna(inplace=True)
    
    transactions = []
    for _, row in customers.iterrows():
        for month in range(int(row['tenure'])):
            transactions.append({
                'transaction_id': f"TXN_{row['customer_id']}_{month:03d}",
                'customer_id': row['customer_id'],
                'month': month + 1,
                'amount': row['monthly_charges']
            })
    
    transactions_df = pd.DataFrame(transactions)
    
    os.makedirs(DATA_RAW, exist_ok=True)
    customers.to_csv(os.path.join(DATA_RAW, 'customers.csv'), index=False)
    transactions_df.to_csv(os.path.join(DATA_RAW, 'transactions.csv'), index=False)
    
    print(f"[OK] Transformed {len(customers)} customers")
    print(f"[OK] Created {len(transactions_df)} transactions")
    print(f"[OK] Churn Rate: {customers['churn'].mean()*100:.2f}%")
    
    return customers, transactions_df

if __name__ == "__main__":
    data = load_telecom_data()
    customers, transactions = transform_to_standard_format(data)
    print("\n[OK] Telecom dataset ready")
