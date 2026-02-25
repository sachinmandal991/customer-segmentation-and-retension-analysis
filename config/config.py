import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# Business KPIs
KPI_CONFIG = {
    'churn_threshold_days': 90,  # Days of inactivity to consider churn
    'high_value_percentile': 80,  # Top 20% customers
    'at_risk_churn_prob': 0.6,   # Churn probability threshold
    'target_retention_rate': 0.85,
    'clv_discount_rate': 0.1
}

# RFM Configuration
RFM_CONFIG = {
    'recency_weight': 0.3,
    'frequency_weight': 0.3,
    'monetary_weight': 0.4
}

# Segmentation Parameters
SEGMENTATION_CONFIG = {
    'kmeans': {
        'n_clusters_range': (3, 8),
        'random_state': 42,
        'n_init': 10,
        'max_iter': 300
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    }
}

# Churn Model Parameters
CHURN_MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'logistic_regression': {
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'scale_pos_weight': 3
    }
}

# Feature Engineering
FEATURE_CONFIG = {
    'transaction_features': [
        'total_transactions',
        'total_spent',
        'avg_transaction_value',
        'days_since_first_purchase',
        'days_since_last_purchase',
        'purchase_frequency'
    ],
    'behavioral_features': [
        'recency',
        'frequency',
        'monetary',
        'rfm_score'
    ]
}

# Segment Definitions
SEGMENT_LABELS = {
    'high_value': 'High-Value Champions',
    'loyal': 'Loyal Customers',
    'at_risk': 'At-Risk Customers',
    'inactive': 'Inactive/Lost',
    'new': 'New Customers',
    'potential': 'Potential Loyalists'
}

# Retention Strategies
RETENTION_STRATEGIES = {
    'high_value': [
        'VIP loyalty program',
        'Exclusive early access',
        'Personal account manager',
        'Premium rewards'
    ],
    'loyal': [
        'Loyalty rewards',
        'Referral bonuses',
        'Special discounts',
        'Engagement campaigns'
    ],
    'at_risk': [
        'Win-back campaigns',
        'Personalized offers',
        'Feedback surveys',
        'Re-engagement emails'
    ],
    'inactive': [
        'Aggressive discounts',
        'Reactivation campaigns',
        'Product recommendations',
        'Limited-time offers'
    ],
    'new': [
        'Onboarding programs',
        'Welcome discounts',
        'Educational content',
        'First purchase incentives'
    ],
    'potential': [
        'Upselling campaigns',
        'Cross-sell opportunities',
        'Engagement programs',
        'Value demonstration'
    ]
}

# Visualization Settings
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'color_palette': 'viridis',
    'style': 'whitegrid',
    'dpi': 300
}
