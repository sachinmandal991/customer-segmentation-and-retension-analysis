import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_RAW, DATA_PROCESSED, KPI_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.reference_date = None
        self.churn_threshold = KPI_CONFIG['churn_threshold_days']
        
    def load_data(self):
        """Load raw customer and transaction data"""
        print("Loading raw data...")
        customers = pd.read_csv(os.path.join(DATA_RAW, 'customers.csv'))
        transactions = pd.read_csv(os.path.join(DATA_RAW, 'transactions.csv'))
        
        print(f"[OK] Loaded {len(customers)} customers and {len(transactions)} transactions")
        return customers, transactions
    
    def clean_data(self, customers, transactions):
        """Clean and validate data"""
        print("Cleaning data...")
        
        customers = customers.drop_duplicates(subset=['customer_id'])
        transactions = transactions.drop_duplicates(subset=['transaction_id'])
        
        valid_customers = set(customers['customer_id'])
        transactions = transactions[transactions['customer_id'].isin(valid_customers)]
        
        print(f"[OK] Cleaned data: {len(customers)} customers, {len(transactions)} transactions")
        return customers, transactions
    
    def create_rfm_features(self, transactions):
        """Create RFM features from telecom data"""
        print("Creating RFM features...")
        
        # For telecom: use tenure as proxy for recency/frequency
        rfm = transactions.groupby('customer_id').agg({
            'month': ['max', 'count'],
            'amount': 'sum'
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Invert recency (higher tenure = lower recency)
        max_tenure = rfm['recency'].max()
        rfm['recency'] = max_tenure - rfm['recency']
        
        # RFM Scores
        rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        rfm['rfm_score'] = (rfm['recency_score'].astype(int) + 
                           rfm['frequency_score'].astype(int) + 
                           rfm['monetary_score'].astype(int))
        
        print(f"[OK] Created RFM features for {len(rfm)} customers")
        return rfm
    
    def create_behavioral_features(self, customers, transactions, rfm):
        """Create behavioral features from telecom data"""
        print("Engineering behavioral features...")
        
        trans_agg = transactions.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean', 'std', 'min', 'max'],
            'month': ['min', 'max']
        }).reset_index()
        
        trans_agg.columns = ['customer_id', 'total_transactions', 'total_spent', 
                            'avg_transaction_value', 'std_transaction_value',
                            'min_transaction_value', 'max_transaction_value',
                            'first_month', 'last_month']
        
        trans_agg['customer_lifetime_days'] = (trans_agg['last_month'] - trans_agg['first_month']) * 30
        trans_agg['days_since_first_purchase'] = trans_agg['last_month'] * 30
        trans_agg['days_since_last_purchase'] = (trans_agg['last_month'].max() - trans_agg['last_month']) * 30
        trans_agg['purchase_frequency'] = trans_agg['total_transactions'] / (trans_agg['customer_lifetime_days'] + 1)
        
        features = customers.merge(trans_agg, on='customer_id', how='left')
        features = features.merge(rfm, on='customer_id', how='left')
        
        # Fill NaN for numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = features[numeric_cols].fillna(0)
        
        features['clv'] = features['total_spent'] * (features['purchase_frequency'] + 1)
        features['is_churned'] = features['churn']
        
        print(f"[OK] Created {len(features.columns)} features for {len(features)} customers")
        return features
    
    def save_processed_data(self, features):
        """Save processed data"""
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        output_path = os.path.join(DATA_PROCESSED, 'customer_features.csv')
        features.to_csv(output_path, index=False)
        print(f"[OK] Saved processed data to {output_path}")
        return output_path

def main():
    preprocessor = DataPreprocessor()
    
    # Load data
    customers, transactions = preprocessor.load_data()
    
    # Clean data
    customers, transactions = preprocessor.clean_data(customers, transactions)
    
    # Create RFM features
    rfm = preprocessor.create_rfm_features(transactions)
    
    # Create behavioral features
    features = preprocessor.create_behavioral_features(customers, transactions, rfm)
    
    # Save processed data
    output_path = preprocessor.save_processed_data(features)
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total Customers: {len(features)}")
    print(f"Total Features: {len(features.columns)}")
    print(f"Churn Rate: {features['is_churned'].mean()*100:.2f}%")
    print(f"Avg CLV: ${features['clv'].mean():.2f}")
    print(f"Output: {output_path}")
    
    return features

if __name__ == "__main__":
    features = main()
