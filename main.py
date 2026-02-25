"""
Main Pipeline Orchestrator
Runs the complete Customer Segmentation and Retention Analysis pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import load_telecom_data, transform_to_standard_format
from src.data.preprocessing import DataPreprocessor
from src.models.segmentation import CustomerSegmentation
from src.models.churn_prediction import ChurnPrediction
from src.models.business_insights import BusinessInsights
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*100)
    print(f"  {text}")
    print("="*100 + "\n")

def main():
    """Run complete pipeline"""
    start_time = time.time()
    
    print_header("CUSTOMER SEGMENTATION AND RETENTION ANALYSIS PIPELINE")
    print("Starting end-to-end pipeline execution...\n")
    
    try:
        # Step 1: Load Telecom Data
        print_header("STEP 1: LOAD TELECOM DATASET")
        data = load_telecom_data()
        customers, transactions = transform_to_standard_format(data)
        
        # Step 2: Data Preprocessing
        print_header("STEP 2: DATA PREPROCESSING")
        preprocessor = DataPreprocessor()
        customers_raw, transactions_raw = preprocessor.load_data()
        customers_clean, transactions_clean = preprocessor.clean_data(customers_raw, transactions_raw)
        rfm = preprocessor.create_rfm_features(transactions_clean)
        features = preprocessor.create_behavioral_features(customers_clean, transactions_clean, rfm)
        preprocessor.save_processed_data(features)
        
        # Step 3: Customer Segmentation
        print_header("STEP 3: CUSTOMER SEGMENTATION")
        segmentation = CustomerSegmentation()
        data = segmentation.load_data()
        X_scaled, feature_cols = segmentation.prepare_features(data)
        optimal_k = segmentation.find_optimal_clusters(X_scaled)
        clusters = segmentation.perform_kmeans_clustering(X_scaled)
        dbscan_labels = segmentation.perform_dbscan_clustering(X_scaled)
        data['outlier'] = (dbscan_labels == -1).astype(int)
        data, segment_profile = segmentation.interpret_segments(data, clusters)
        segmentation.visualize_segments(data, X_scaled)
        segmentation.save_models()
        
        from config.config import DATA_PROCESSED
        data.to_csv(os.path.join(DATA_PROCESSED, 'customer_segments.csv'), index=False)
        segment_profile.to_csv(os.path.join(DATA_PROCESSED, 'segment_profiles.csv'))
        
        # Step 4: Churn Prediction
        print_header("STEP 4: CHURN PREDICTION")
        churn_model = ChurnPrediction()
        data = churn_model.load_data()
        X, y, feature_cols = churn_model.prepare_features(data)
        churn_model.feature_names = feature_cols
        X_train, X_test, y_train, y_test = churn_model.split_data(X, y)
        X_train_balanced, y_train_balanced = churn_model.handle_imbalance(X_train, y_train)
        churn_model.train_models(X_train_balanced, y_train_balanced, X_test, y_test)
        churn_model.visualize_results(y_test)
        churn_model.save_models()
        data = churn_model.predict_churn_probabilities(data, X)
        
        # Step 5: Business Insights
        print_header("STEP 5: BUSINESS INSIGHTS GENERATION")
        insights_engine = BusinessInsights()
        data = insights_engine.load_data()
        kpis = insights_engine.calculate_kpis(data)
        segment_insights = insights_engine.segment_analysis(data)
        recommendations = insights_engine.generate_recommendations(data)
        top_priority = insights_engine.customer_prioritization(data)
        insights_engine.create_executive_dashboard(data, kpis, segment_insights)
        insights_engine.generate_report(data, kpis, segment_insights, recommendations, top_priority)
        
        # Pipeline Complete
        elapsed_time = time.time() - start_time
        
        print_header("PIPELINE EXECUTION COMPLETE")
        print(f"[OK] Total execution time: {elapsed_time:.2f} seconds")
        print(f"[OK] All outputs saved successfully")
        print(f"\nNext Steps:")
        print(f"  1. Review outputs in 'outputs/' directory")
        print(f"  2. Check visualizations in 'outputs/figures/'")
        print(f"  3. Read business report in 'outputs/reports/'")
        print(f"  4. Launch dashboard: streamlit run dashboard/app.py")
        print("\n" + "="*100 + "\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
