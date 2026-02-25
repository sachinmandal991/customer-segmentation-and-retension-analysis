import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_RAW

def load_telecom_data():
    """Load Telecom Customer Churn Dataset"""
    print("Loading Telecom Customer Churn Dataset...")
    
    # Download from Kaggle or use local file
    # URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    
    try:
        # Try to load from raw folder
        data = pd.read_csv(os.path.join(DATA_RAW, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'))
        print(f"✓ Loaded {len(data)} customers from local file")
    except:
        print("⚠️ Dataset not found. Please download from:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print(f"Save as: {os.path.join(DATA_RAW, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')}")
        
        # Create sample instructions
        os.makedirs(DATA_RAW, exist_ok=True)
        with open(os.path.join(DATA_RAW, 'DOWNLOAD_INSTRUCTIONS.txt'), 'w') as f:
            f.write("TELECOM DATASET DOWNLOAD INSTRUCTIONS\n")
            f.write("="*50 + "\n\n")
            f.write("1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n")
            f.write("2. Download: WA_Fn-UseC_-Telco-Customer-Churn.csv\n")
            f.write(f"3. Place in: {DATA_RAW}\n")
            f.write("4. Run: python src/data/telecom_loader.py\n")
        
        sys.exit(1)
    
    return data

def transform_to_standard_format(data):
    """Transform telecom data to standard format"""
    print("Transforming data to standard format...")
    
    # Create customer table
    customers = pd.DataFrame({
        'customer_id': data['customerID'],
        'gender': data['gender'],
        'senior_citizen': data['SeniorCitizen'],
        'partner': data['Partner'],
        'dependents': data['Dependents'],
        'tenure': data['tenure'],
        'phone_service': data['PhoneService'],
        'multiple_lines': data['MultipleLines'],
        'internet_service': data['InternetService'],
        'online_security': data['OnlineSecurity'],
        'online_backup': data['OnlineBackup'],
        'device_protection': data['DeviceProtection'],
        'tech_support': data['TechSupport'],
        'streaming_tv': data['StreamingTV'],
        'streaming_movies': data['StreamingMovies'],
        'contract': data['Contract'],
        'paperless_billing': data['PaperlessBilling'],
        'payment_method': data['PaymentMethod'],
        'monthly_charges': pd.to_numeric(data['MonthlyCharges'], errors='coerce'),
        'total_charges': pd.to_numeric(data['TotalCharges'], errors='coerce'),
        'churn': data['Churn'].map({'Yes': 1, 'No': 0})
    })
    
    # Clean data
    customers['total_charges'].fillna(customers['monthly_charges'] * customers['tenure'], inplace=True)
    customers.dropna(inplace=True)
    
    # Create transaction-like data from tenure and charges
    transactions = []
    for _, row in customers.iterrows():
        tenure_months = int(row['tenure'])
        if tenure_months > 0:
            for month in range(tenure_months):
                transactions.append({
                    'transaction_id': f"TXN_{row['customer_id']}_{month:03d}",
                    'customer_id': row['customer_id'],
                    'month': month + 1,
                    'amount': row['monthly_charges'],
                    'service_type': row['internet_service']
                })
    
    transactions_df = pd.DataFrame(transactions)
    
    # Save
    os.makedirs(DATA_RAW, exist_ok=True)
    customers.to_csv(os.path.join(DATA_RAW, 'telecom_customers.csv'), index=False)
    transactions_df.to_csv(os.path.join(DATA_RAW, 'telecom_transactions.csv'), index=False)
    
    print(f"✓ Transformed {len(customers)} customers")
    print(f"✓ Created {len(transactions_df)} transaction records")
    print(f"✓ Churn Rate: {customers['churn'].mean()*100:.2f}%")
    
    return customers, transactions_df

if __name__ == "__main__":
    data = load_telecom_data()
    customers, transactions = transform_to_standard_format(data)
    print("\n✓ Telecom dataset ready for processing")
