import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, SEGMENTATION_CONFIG, SEGMENT_LABELS

class CustomerSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.optimal_k = None
        
    def load_data(self):
        """Load processed customer features"""
        print("Loading processed data...")
        data = pd.read_csv(os.path.join(DATA_PROCESSED, 'customer_features.csv'))
        print(f"[OK] Loaded {len(data)} customers")
        return data
    
    def prepare_features(self, data):
        """Select and normalize features for clustering"""
        print("Preparing features for segmentation...")
        
        # Select key features for segmentation
        feature_cols = ['recency', 'frequency', 'monetary', 'total_transactions',
                       'avg_transaction_value', 'customer_lifetime_days', 
                       'purchase_frequency', 'clv']
        
        X = data[feature_cols].copy()
        X.fillna(0, inplace=True)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"[OK] Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} customers")
        return X_scaled, feature_cols
    
    def find_optimal_clusters(self, X_scaled):
        """Find optimal number of clusters using Elbow and Silhouette methods"""
        print("Finding optimal number of clusters...")
        
        k_range = range(SEGMENTATION_CONFIG['kmeans']['n_clusters_range'][0],
                       SEGMENTATION_CONFIG['kmeans']['n_clusters_range'][1] + 1)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, 
                          random_state=SEGMENTATION_CONFIG['kmeans']['random_state'],
                          n_init=SEGMENTATION_CONFIG['kmeans']['n_init'])
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Plot Elbow and Silhouette
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'optimal_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Select optimal k (highest silhouette score)
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"[OK] Optimal number of clusters: {self.optimal_k}")
        print(f"[OK] Silhouette Score: {max(silhouette_scores):.3f}")
        
        return self.optimal_k
    
    def perform_kmeans_clustering(self, X_scaled):
        """Perform KMeans clustering"""
        print(f"Performing KMeans clustering with k={self.optimal_k}...")
        
        self.kmeans_model = KMeans(
            n_clusters=self.optimal_k,
            random_state=SEGMENTATION_CONFIG['kmeans']['random_state'],
            n_init=SEGMENTATION_CONFIG['kmeans']['n_init'],
            max_iter=SEGMENTATION_CONFIG['kmeans']['max_iter']
        )
        
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Evaluation metrics
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        
        print(f"[OK] KMeans Clustering Complete")
        print(f"  - Silhouette Score: {silhouette:.3f}")
        print(f"  - Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        return clusters
    
    def perform_dbscan_clustering(self, X_scaled):
        """Perform DBSCAN for outlier detection"""
        print("Performing DBSCAN for outlier detection...")
        
        self.dbscan_model = DBSCAN(
            eps=SEGMENTATION_CONFIG['dbscan']['eps'],
            min_samples=SEGMENTATION_CONFIG['dbscan']['min_samples']
        )
        
        dbscan_labels = self.dbscan_model.fit_predict(X_scaled)
        
        n_outliers = np.sum(dbscan_labels == -1)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        print(f"[OK] DBSCAN Complete")
        print(f"  - Clusters found: {n_clusters}")
        print(f"  - Outliers detected: {n_outliers} ({n_outliers/len(dbscan_labels)*100:.2f}%)")
        
        return dbscan_labels
    
    def interpret_segments(self, data, clusters):
        """Interpret clusters into business segments"""
        print("Interpreting segments...")
        
        data['segment'] = clusters
        
        # Analyze segment characteristics
        segment_profile = data.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'clv': 'mean',
            'total_spent': 'sum',
            'is_churned': 'mean'
        }).round(2)
        
        segment_profile.columns = ['count', 'avg_recency', 'avg_frequency', 
                                   'avg_monetary', 'avg_clv', 'total_revenue', 'churn_rate']
        
        # Assign business labels based on characteristics
        segment_names = {}
        for seg in segment_profile.index:
            profile = segment_profile.loc[seg]
            
            if profile['avg_monetary'] > segment_profile['avg_monetary'].quantile(0.75) and profile['avg_recency'] < 60:
                segment_names[seg] = 'High-Value Champions'
            elif profile['avg_frequency'] > segment_profile['avg_frequency'].median() and profile['avg_recency'] < 90:
                segment_names[seg] = 'Loyal Customers'
            elif profile['avg_recency'] > 90 and profile['churn_rate'] > 0.5:
                segment_names[seg] = 'At-Risk Customers'
            elif profile['avg_recency'] > 180:
                segment_names[seg] = 'Inactive/Lost'
            elif profile['avg_frequency'] < segment_profile['avg_frequency'].quantile(0.25):
                segment_names[seg] = 'New Customers'
            else:
                segment_names[seg] = 'Potential Loyalists'
        
        data['segment_name'] = data['segment'].map(segment_names)
        segment_profile['segment_name'] = segment_profile.index.map(segment_names)
        
        print("\n" + "="*80)
        print("SEGMENT PROFILES")
        print("="*80)
        print(segment_profile.to_string())
        print("="*80)
        
        return data, segment_profile
    
    def visualize_segments(self, data, X_scaled):
        """Create segment visualizations"""
        print("Creating visualizations...")
        
        from sklearn.decomposition import PCA
        
        # PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. PCA Scatter Plot
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                     c=data['segment'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0, 0].set_title('Customer Segments (PCA)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Segment Distribution
        segment_counts = data['segment_name'].value_counts()
        axes[0, 1].bar(range(len(segment_counts)), segment_counts.values, color='skyblue')
        axes[0, 1].set_xticks(range(len(segment_counts)))
        axes[0, 1].set_xticklabels(segment_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].set_title('Segment Distribution')
        
        # 3. Revenue by Segment
        revenue_by_segment = data.groupby('segment_name')['total_spent'].sum().sort_values(ascending=False)
        axes[1, 0].barh(range(len(revenue_by_segment)), revenue_by_segment.values, color='green', alpha=0.7)
        axes[1, 0].set_yticks(range(len(revenue_by_segment)))
        axes[1, 0].set_yticklabels(revenue_by_segment.index)
        axes[1, 0].set_xlabel('Total Revenue ($)')
        axes[1, 0].set_title('Revenue Contribution by Segment')
        
        # 4. RFM Heatmap by Segment
        rfm_by_segment = data.groupby('segment_name')[['recency', 'frequency', 'monetary']].mean()
        sns.heatmap(rfm_by_segment.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('RFM Profile by Segment')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'segment_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualizations saved to {FIGURES_DIR}")
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
        joblib.dump(self.kmeans_model, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
        joblib.dump(self.dbscan_model, os.path.join(MODELS_DIR, 'dbscan_model.pkl'))
        
        print(f"[OK] Models saved to {MODELS_DIR}")

def main():
    segmentation = CustomerSegmentation()
    
    # Load data
    data = segmentation.load_data()
    
    # Prepare features
    X_scaled, feature_cols = segmentation.prepare_features(data)
    
    # Find optimal clusters
    optimal_k = segmentation.find_optimal_clusters(X_scaled)
    
    # Perform KMeans clustering
    clusters = segmentation.perform_kmeans_clustering(X_scaled)
    
    # Perform DBSCAN
    dbscan_labels = segmentation.perform_dbscan_clustering(X_scaled)
    data['outlier'] = (dbscan_labels == -1).astype(int)
    
    # Interpret segments
    data, segment_profile = segmentation.interpret_segments(data, clusters)
    
    # Visualize segments
    segmentation.visualize_segments(data, X_scaled)
    
    # Save results
    data.to_csv(os.path.join(DATA_PROCESSED, 'customer_segments.csv'), index=False)
    segment_profile.to_csv(os.path.join(DATA_PROCESSED, 'segment_profiles.csv'))
    
    # Save models
    segmentation.save_models()
    
    print("\n" + "="*80)
    print("SEGMENTATION COMPLETE")
    print("="*80)
    
    return data, segment_profile

if __name__ == "__main__":
    data, segment_profile = main()
