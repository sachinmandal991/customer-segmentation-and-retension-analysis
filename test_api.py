import requests
import json

# API Base URL
BASE_URL = "http://localhost:5000"

print("="*60)
print("Testing Customer Analytics API")
print("="*60)

# Test 1: Get API Info
print("\n1. Testing API Info...")
response = requests.get(f"{BASE_URL}/")
print(json.dumps(response.json(), indent=2))

# Test 2: Get Statistics
print("\n2. Testing Statistics...")
response = requests.get(f"{BASE_URL}/api/stats")
stats = response.json()
if stats['success']:
    print(f"Total Customers: {stats['stats']['total_customers']:,}")
    print(f"Retention Rate: {stats['stats']['retention_rate']:.2f}%")
    print(f"Churn Rate: {stats['stats']['churn_rate']:.2f}%")
    print(f"Total Revenue: ${stats['stats']['total_revenue']:,.2f}")

# Test 3: Predict Churn
print("\n3. Testing Churn Prediction...")
customer_data = {
    "recency": 30,
    "frequency": 10,
    "monetary": 500,
    "total_transactions": 10,
    "avg_transaction_value": 50,
    "customer_lifetime_days": 365,
    "purchase_frequency": 0.027,
    "clv": 1000,
    "rfm_score": 12,
    "tenure": 24,
    "monthly_charges": 70
}
response = requests.post(f"{BASE_URL}/api/predict", json=customer_data)
result = response.json()
if result['success']:
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")

# Test 4: Predict Segment
print("\n4. Testing Segment Prediction...")
segment_data = {
    "recency": 15,
    "frequency": 50,
    "monetary": 4000,
    "total_transactions": 50,
    "avg_transaction_value": 80,
    "customer_lifetime_days": 730,
    "purchase_frequency": 0.068,
    "clv": 5000
}
response = requests.post(f"{BASE_URL}/api/segment", json=segment_data)
result = response.json()
if result['success']:
    print(f"Segment: {result['segment_name']}")
    print(f"Recommendation: {result['recommendation']}")

# Test 5: Get Segments
print("\n5. Testing Segment Analysis...")
response = requests.get(f"{BASE_URL}/api/segments")
segments = response.json()
if segments['success']:
    print("\nSegment Summary:")
    for seg in segments['segments']:
        print(f"  {seg['segment_name']}: {seg['customer_count']} customers, ${seg['total_revenue']:,.0f} revenue")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
