from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
MODEL_DIR = 'models'
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
kmeans_model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
churn_model = joblib.load(os.path.join(MODEL_DIR, 'best_churn_model.pkl'))

# Load processed data
DATA_DIR = 'data/processed'
customers_df = pd.read_csv(os.path.join(DATA_DIR, 'customer_churn_predictions.csv'))

@app.route('/')
def home():
    """API Home"""
    return jsonify({
        'message': 'Customer Segmentation & Churn Prediction API',
        'version': '1.0',
        'endpoints': {
            '/api/predict': 'POST - Predict churn for a customer',
            '/api/segment': 'POST - Get customer segment',
            '/api/customers': 'GET - Get all customers',
            '/api/customer/<id>': 'GET - Get specific customer',
            '/api/stats': 'GET - Get overall statistics',
            '/api/segments': 'GET - Get segment analysis'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_churn():
    """Predict churn probability for a customer"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'success': False, 'error': 'Missing customer_id'}), 400
        
        customer = customers_df[customers_df['customer_id'] == customer_id]
        if customer.empty:
            return jsonify({'success': False, 'error': 'Customer not found'}), 404
        
        customer_data = customer.iloc[0]
        
        return jsonify({
            'customer_id': customer_id,
            'churn_probability': float(customer_data['churn_probability']),
            'churn_prediction': 'Yes' if customer_data['is_churned'] == 1 else 'No',
            'risk_level': customer_data['churn_risk'],
            'segment': customer_data['segment_name']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/segment', methods=['POST'])
def predict_segment():
    """Predict customer segment"""
    try:
        data = request.json
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'success': False, 'error': 'Missing customer_id'}), 400
        
        customer = customers_df[customers_df['customer_id'] == customer_id]
        if customer.empty:
            return jsonify({'success': False, 'error': 'Customer not found'}), 404
        
        customer_data = customer.iloc[0]
        
        return jsonify({
            'customer_id': customer_id,
            'segment': customer_data['segment_name'],
            'segment_id': int(customer_data['segment']),
            'rfm_score': float(customer_data['rfm_score']),
            'clv': float(customer_data['clv'])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/customers', methods=['GET'])
def get_customers():
    """Get all customers with pagination"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 10)), 100)
        
        start = (page - 1) * per_page
        end = start + per_page
        
        customers_subset = customers_df.iloc[start:end]
        customers = [{
            'customer_id': row['customer_id'],
            'segment': row['segment_name'],
            'churn_probability': float(row['churn_probability']),
            'clv': float(row['clv'])
        } for _, row in customers_subset.iterrows()]
        
        return jsonify({
            'page': page,
            'per_page': per_page,
            'total': len(customers_df),
            'total_pages': (len(customers_df) + per_page - 1) // per_page,
            'customers': customers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/customer/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    """Get specific customer details"""
    try:
        customer = customers_df[customers_df['customer_id'] == customer_id]
        
        if customer.empty:
            return jsonify({'error': 'Customer not found'}), 404
        
        customer_data = customer.iloc[0]
        return jsonify({
            'customer_id': customer_id,
            'segment': customer_data['segment_name'],
            'churn_probability': float(customer_data['churn_probability']),
            'churn_prediction': 'Yes' if customer_data['is_churned'] == 1 else 'No',
            'risk_level': customer_data['churn_risk'],
            'clv': float(customer_data['clv']),
            'rfm_score': float(customer_data['rfm_score']),
            'tenure': float(customer_data['tenure']),
            'monthly_charges': float(customer_data['monthly_charges']),
            'total_charges': float(customer_data['total_spent'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    try:
        return jsonify({
            'total_customers': int(len(customers_df)),
            'churn_rate': float(customers_df['is_churned'].mean() * 100),
            'retention_rate': float((1 - customers_df['is_churned'].mean()) * 100),
            'total_revenue': float(customers_df['total_spent'].sum()),
            'avg_clv': float(customers_df['clv'].mean()),
            'high_risk_customers': int(len(customers_df[customers_df['churn_risk'] == 'High']))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/segments', methods=['GET'])
def get_segments():
    """Get segment analysis"""
    try:
        total_revenue = customers_df['total_spent'].sum()
        segments = []
        
        for segment_name in customers_df['segment_name'].unique():
            segment_data = customers_df[customers_df['segment_name'] == segment_name]
            segment_revenue = segment_data['total_spent'].sum()
            
            segments.append({
                'segment_name': segment_name,
                'segment_id': int(segment_data['segment'].iloc[0]),
                'customer_count': int(len(segment_data)),
                'avg_clv': float(segment_data['clv'].mean()),
                'churn_rate': float(segment_data['is_churned'].mean() * 100),
                'revenue_contribution': float((segment_revenue / total_revenue) * 100)
            })
        
        return jsonify({'segments': segments})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendation(risk_level):
    """Get retention recommendation based on risk level"""
    recommendations = {
        'High': 'Immediate action required: Win-back campaign, personalized offers, feedback survey',
        'Medium': 'Monitor closely: Engagement campaigns, loyalty rewards, special discounts',
        'Low': 'Maintain relationship: Regular communication, referral bonuses, value demonstration'
    }
    return recommendations.get(risk_level, 'Continue monitoring')

def get_segment_recommendation(segment_name):
    """Get recommendation based on segment"""
    recommendations = {
        'High-Value Champions': 'VIP loyalty program, exclusive early access, personal account manager',
        'Loyal Customers': 'Loyalty rewards, referral bonuses, special discounts',
        'At-Risk Customers': 'Win-back campaigns, personalized offers, feedback surveys',
        'Inactive/Lost': 'Aggressive discounts, reactivation campaigns, limited-time offers',
        'New Customers': 'Onboarding programs, welcome discounts, educational content',
        'Potential Loyalists': 'Upselling campaigns, cross-sell opportunities, engagement programs'
    }
    return recommendations.get(segment_name, 'Standard engagement')

if __name__ == '__main__':
    print("="*60)
    print("Customer Analytics API Server")
    print("="*60)
    print("API running on: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                      - API info")
    print("  POST /api/predict           - Predict churn")
    print("  POST /api/segment           - Predict segment")
    print("  GET  /api/customers         - Get all customers")
    print("  GET  /api/customer/<id>     - Get customer by ID")
    print("  GET  /api/stats             - Get statistics")
    print("  GET  /api/segments          - Get segment analysis")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
