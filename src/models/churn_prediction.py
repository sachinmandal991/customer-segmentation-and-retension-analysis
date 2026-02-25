import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, f1_score)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_PROCESSED, MODELS_DIR, FIGURES_DIR, CHURN_MODEL_CONFIG

class ChurnPrediction:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load segmented customer data"""
        print("Loading customer data...")
        data = pd.read_csv(os.path.join(DATA_PROCESSED, 'customer_segments.csv'))
        print(f"[OK] Loaded {len(data)} customers")
        return data
    
    def prepare_features(self, data):
        """Prepare features for churn prediction"""
        print("Preparing features for churn prediction...")
        
        # Select only numeric features
        feature_cols = ['recency', 'frequency', 'monetary', 'total_transactions',
                       'avg_transaction_value', 'customer_lifetime_days',
                       'purchase_frequency', 'clv', 'rfm_score', 'tenure', 'monthly_charges']
        
        # Filter to only existing columns
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        X = data[feature_cols].copy()
        y = data['is_churned'].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        print(f"[OK] Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"[OK] Churn Rate: {y.mean()*100:.2f}%")
        
        return X, y, feature_cols
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=CHURN_MODEL_CONFIG['test_size'],
            random_state=CHURN_MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        print("Handling class imbalance with SMOTE...")
        
        smote = SMOTE(random_state=CHURN_MODEL_CONFIG['random_state'])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"[OK] Original: {len(y_train)}, Balanced: {len(y_train_balanced)}")
        print(f"[OK] Churn distribution: {np.bincount(y_train_balanced)}")
        
        return X_train_balanced, y_train_balanced
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple classification models"""
        print("\nTraining churn prediction models...")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(
                **CHURN_MODEL_CONFIG['logistic_regression'],
                random_state=CHURN_MODEL_CONFIG['random_state']
            ),
            'Random Forest': RandomForestClassifier(
                **CHURN_MODEL_CONFIG['random_forest'],
                random_state=CHURN_MODEL_CONFIG['random_state']
            ),
            'XGBoost': XGBClassifier(
                **CHURN_MODEL_CONFIG['xgboost'],
                random_state=CHURN_MODEL_CONFIG['random_state'],
                eval_metric='logloss'
            )
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print('='*60)
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=CHURN_MODEL_CONFIG['cv_folds'], 
                                       scoring='roc_auc')
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"[OK] ROC-AUC: {roc_auc:.4f}")
            print(f"[OK] F1 Score: {f1:.4f}")
            print(f"[OK] CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
        
        # Select best model
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"Best Model: {self.best_model_name}")
        print(f"ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")
        print('='*60)
    
    def visualize_results(self, y_test):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. ROC Curves
        ax1 = plt.subplot(2, 3, 1)
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Precision-Recall Curves
        ax2 = plt.subplot(2, 3, 2)
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
            ax2.plot(recall, precision, label=name)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Model Comparison
        ax3 = plt.subplot(2, 3, 3)
        model_names = list(self.results.keys())
        roc_scores = [self.results[name]['roc_auc'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        ax3.bar(x - width/2, roc_scores, width, label='ROC-AUC', alpha=0.8)
        ax3.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, axis='y')
        
        # 4. Confusion Matrix (Best Model)
        ax4 = plt.subplot(2, 3, 4)
        cm = confusion_matrix(y_test, self.results[self.best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title(f'Confusion Matrix - {self.best_model_name}')
        
        # 5. Feature Importance (Best Model)
        ax5 = plt.subplot(2, 3, 5)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            ax5.barh(range(len(indices)), importances[indices], color='teal')
            ax5.set_yticks(range(len(indices)))
            ax5.set_yticklabels([self.feature_names[i] for i in indices])
            ax5.set_xlabel('Importance')
            ax5.set_title(f'Top 10 Features - {self.best_model_name}')
        
        # 6. Churn Probability Distribution
        ax6 = plt.subplot(2, 3, 6)
        churn_probs = self.results[self.best_model_name]['y_pred_proba']
        ax6.hist(churn_probs[y_test == 0], bins=30, alpha=0.6, label='Not Churned', color='green')
        ax6.hist(churn_probs[y_test == 1], bins=30, alpha=0.6, label='Churned', color='red')
        ax6.set_xlabel('Churn Probability')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Churn Probability Distribution')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'churn_model_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualizations saved to {FIGURES_DIR}")
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for name, result in self.results.items():
            model_filename = name.lower().replace(' ', '_') + '_model.pkl'
            joblib.dump(result['model'], os.path.join(MODELS_DIR, model_filename))
        
        # Save best model separately
        joblib.dump(self.best_model, os.path.join(MODELS_DIR, 'best_churn_model.pkl'))
        
        print(f"[OK] Models saved to {MODELS_DIR}")
    
    def predict_churn_probabilities(self, data, X):
        """Generate churn probabilities for all customers"""
        print("\nGenerating churn probabilities...")
        
        churn_probs = self.best_model.predict_proba(X)[:, 1]
        data['churn_probability'] = churn_probs
        data['churn_risk'] = pd.cut(churn_probs, bins=[0, 0.3, 0.6, 1.0], 
                                    labels=['Low', 'Medium', 'High'])
        
        # Save results
        output_path = os.path.join(DATA_PROCESSED, 'customer_churn_predictions.csv')
        data.to_csv(output_path, index=False)
        
        print(f"[OK] Churn probabilities saved to {output_path}")
        print(f"\nChurn Risk Distribution:")
        print(data['churn_risk'].value_counts())
        
        return data

def main():
    churn_model = ChurnPrediction()
    
    # Load data
    data = churn_model.load_data()
    
    # Prepare features
    X, y, feature_cols = churn_model.prepare_features(data)
    churn_model.feature_names = feature_cols
    
    # Split data
    X_train, X_test, y_train, y_test = churn_model.split_data(X, y)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = churn_model.handle_imbalance(X_train, y_train)
    
    # Train models
    churn_model.train_models(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Visualize results
    churn_model.visualize_results(y_test)
    
    # Save models
    churn_model.save_models()
    
    # Predict churn probabilities for all customers
    data = churn_model.predict_churn_probabilities(data, X)
    
    print("\n" + "="*80)
    print("CHURN PREDICTION COMPLETE")
    print("="*80)
    
    return data, churn_model.results

if __name__ == "__main__":
    data, results = main()
