import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (DATA_PROCESSED, REPORTS_DIR, FIGURES_DIR, 
                          RETENTION_STRATEGIES, KPI_CONFIG)

class BusinessInsights:
    def __init__(self):
        self.insights = {}
        
    def load_data(self):
        """Load customer data with segments and churn predictions"""
        print("Loading customer data...")
        data = pd.read_csv(os.path.join(DATA_PROCESSED, 'customer_churn_predictions.csv'))
        print(f"[OK] Loaded {len(data)} customers")
        return data
    
    def calculate_kpis(self, data):
        """Calculate business KPIs"""
        print("\nCalculating Business KPIs...")
        
        kpis = {
            'total_customers': len(data),
            'active_customers': len(data[data['is_churned'] == 0]),
            'churned_customers': len(data[data['is_churned'] == 1]),
            'retention_rate': (1 - data['is_churned'].mean()) * 100,
            'churn_rate': data['is_churned'].mean() * 100,
            'total_revenue': data['total_spent'].sum(),
            'avg_clv': data['clv'].mean(),
            'avg_transaction_value': data['avg_transaction_value'].mean(),
            'high_risk_customers': len(data[data['churn_risk'] == 'High']),
            'revenue_at_risk': data[data['churn_risk'] == 'High']['total_spent'].sum()
        }
        
        self.insights['kpis'] = kpis
        
        print("\n" + "="*80)
        print("KEY PERFORMANCE INDICATORS")
        print("="*80)
        print(f"Total Customers: {kpis['total_customers']:,}")
        print(f"Active Customers: {kpis['active_customers']:,}")
        print(f"Churned Customers: {kpis['churned_customers']:,}")
        print(f"Retention Rate: {kpis['retention_rate']:.2f}%")
        print(f"Churn Rate: {kpis['churn_rate']:.2f}%")
        print(f"Total Revenue: ${kpis['total_revenue']:,.2f}")
        print(f"Average CLV: ${kpis['avg_clv']:,.2f}")
        print(f"High Risk Customers: {kpis['high_risk_customers']:,}")
        print(f"Revenue at Risk: ${kpis['revenue_at_risk']:,.2f}")
        print("="*80)
        
        return kpis
    
    def segment_analysis(self, data):
        """Analyze segments with business insights"""
        print("\nPerforming Segment Analysis...")
        
        segment_insights = data.groupby('segment_name').agg({
            'customer_id': 'count',
            'total_spent': ['sum', 'mean'],
            'clv': 'mean',
            'is_churned': 'mean',
            'churn_probability': 'mean',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        segment_insights.columns = ['customer_count', 'total_revenue', 'avg_revenue',
                                   'avg_clv', 'churn_rate', 'avg_churn_prob',
                                   'avg_recency', 'avg_frequency', 'avg_monetary']
        
        # Calculate revenue contribution
        segment_insights['revenue_contribution_%'] = (
            segment_insights['total_revenue'] / segment_insights['total_revenue'].sum() * 100
        ).round(2)
        
        # Priority score (higher CLV, lower churn = higher priority)
        segment_insights['priority_score'] = (
            segment_insights['avg_clv'] / segment_insights['avg_clv'].max() * 0.6 +
            (1 - segment_insights['churn_rate']) * 0.4
        ).round(3)
        
        segment_insights = segment_insights.sort_values('priority_score', ascending=False)
        
        self.insights['segments'] = segment_insights
        
        print("\n" + "="*80)
        print("SEGMENT ANALYSIS")
        print("="*80)
        print(segment_insights.to_string())
        print("="*80)
        
        return segment_insights
    
    def generate_recommendations(self, data):
        """Generate actionable retention recommendations"""
        print("\nGenerating Retention Recommendations...")
        
        recommendations = []
        
        for segment in data['segment_name'].unique():
            segment_data = data[data['segment_name'] == segment]
            
            # Map segment to strategy category
            if 'High-Value' in segment or 'Champions' in segment:
                strategy_key = 'high_value'
            elif 'Loyal' in segment:
                strategy_key = 'loyal'
            elif 'At-Risk' in segment:
                strategy_key = 'at_risk'
            elif 'Inactive' in segment or 'Lost' in segment:
                strategy_key = 'inactive'
            elif 'New' in segment:
                strategy_key = 'new'
            else:
                strategy_key = 'potential'
            
            # Get strategies
            strategies = RETENTION_STRATEGIES.get(strategy_key, [])
            
            # Calculate impact metrics
            customer_count = len(segment_data)
            revenue = segment_data['total_spent'].sum()
            avg_churn_prob = segment_data['churn_probability'].mean()
            high_risk_count = len(segment_data[segment_data['churn_risk'] == 'High'])
            
            recommendations.append({
                'segment': segment,
                'customer_count': customer_count,
                'total_revenue': revenue,
                'avg_churn_probability': avg_churn_prob,
                'high_risk_customers': high_risk_count,
                'recommended_actions': ', '.join(strategies[:3]),
                'priority': 'High' if avg_churn_prob > 0.6 else 'Medium' if avg_churn_prob > 0.3 else 'Low'
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('avg_churn_probability', ascending=False)
        
        self.insights['recommendations'] = recommendations_df
        
        print("\n" + "="*80)
        print("RETENTION RECOMMENDATIONS")
        print("="*80)
        print(recommendations_df.to_string(index=False))
        print("="*80)
        
        return recommendations_df
    
    def customer_prioritization(self, data):
        """Prioritize customers for retention efforts"""
        print("\nPrioritizing Customers...")
        
        # Create priority score
        data['retention_priority'] = (
            data['clv'] / data['clv'].max() * 0.4 +
            data['churn_probability'] * 0.4 +
            (data['total_spent'] / data['total_spent'].max()) * 0.2
        )
        
        # Get top priority customers
        top_priority = data.nlargest(100, 'retention_priority')[
            ['customer_id', 'segment_name', 'churn_probability', 'churn_risk', 
             'clv', 'total_spent', 'retention_priority']
        ].round(3)
        
        self.insights['top_priority_customers'] = top_priority
        
        print(f"[OK] Identified top 100 priority customers")
        print(f"  - Average CLV: ${top_priority['clv'].mean():,.2f}")
        print(f"  - Total Revenue: ${top_priority['total_spent'].sum():,.2f}")
        print(f"  - High Risk: {len(top_priority[top_priority['churn_risk'] == 'High'])}")
        
        return top_priority
    
    def create_executive_dashboard(self, data, kpis, segment_insights):
        """Create executive summary visualizations"""
        print("\nCreating Executive Dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. KPI Summary
        ax1 = plt.subplot(3, 3, 1)
        kpi_labels = ['Retention\nRate', 'Churn\nRate', 'Avg CLV']
        kpi_values = [kpis['retention_rate'], kpis['churn_rate'], kpis['avg_clv']/100]
        colors = ['green', 'red', 'blue']
        ax1.bar(kpi_labels, kpi_values, color=colors, alpha=0.7)
        ax1.set_title('Key Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value')
        
        # 2. Revenue by Segment
        ax2 = plt.subplot(3, 3, 2)
        segment_revenue = data.groupby('segment_name')['total_spent'].sum().sort_values(ascending=False)
        ax2.pie(segment_revenue.values, labels=segment_revenue.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Revenue Distribution by Segment', fontsize=14, fontweight='bold')
        
        # 3. Churn Risk Distribution
        ax3 = plt.subplot(3, 3, 3)
        churn_risk_counts = data['churn_risk'].value_counts()
        colors_risk = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        ax3.bar(churn_risk_counts.index, churn_risk_counts.values, 
               color=[colors_risk[x] for x in churn_risk_counts.index], alpha=0.7)
        ax3.set_title('Churn Risk Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Customers')
        
        # 4. Customer Count by Segment
        ax4 = plt.subplot(3, 3, 4)
        segment_counts = data['segment_name'].value_counts()
        ax4.barh(range(len(segment_counts)), segment_counts.values, color='skyblue')
        ax4.set_yticks(range(len(segment_counts)))
        ax4.set_yticklabels(segment_counts.index)
        ax4.set_xlabel('Number of Customers')
        ax4.set_title('Customers by Segment', fontsize=14, fontweight='bold')
        
        # 5. CLV Distribution
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(data['clv'], bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax5.axvline(data['clv'].mean(), color='red', linestyle='--', label=f'Mean: ${data["clv"].mean():.0f}')
        ax5.set_xlabel('Customer Lifetime Value ($)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('CLV Distribution', fontsize=14, fontweight='bold')
        ax5.legend()
        
        # 6. Segment Performance Matrix
        ax6 = plt.subplot(3, 3, 6)
        segment_matrix = data.groupby('segment_name').agg({
            'clv': 'mean',
            'churn_probability': 'mean'
        })
        scatter = ax6.scatter(segment_matrix['churn_probability'], segment_matrix['clv'], 
                            s=200, alpha=0.6, c=range(len(segment_matrix)), cmap='viridis')
        for idx, segment in enumerate(segment_matrix.index):
            ax6.annotate(segment, (segment_matrix['churn_probability'].iloc[idx], 
                                  segment_matrix['clv'].iloc[idx]), fontsize=8)
        ax6.set_xlabel('Average Churn Probability')
        ax6.set_ylabel('Average CLV ($)')
        ax6.set_title('Segment Performance Matrix', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Revenue at Risk
        ax7 = plt.subplot(3, 3, 7)
        risk_revenue = data.groupby('churn_risk')['total_spent'].sum()
        ax7.bar(risk_revenue.index, risk_revenue.values, 
               color=[colors_risk[x] for x in risk_revenue.index], alpha=0.7)
        ax7.set_ylabel('Total Revenue ($)')
        ax7.set_title('Revenue at Risk', fontsize=14, fontweight='bold')
        
        # 8. Churn Rate by Segment
        ax8 = plt.subplot(3, 3, 8)
        churn_by_segment = data.groupby('segment_name')['is_churned'].mean().sort_values(ascending=False) * 100
        ax8.barh(range(len(churn_by_segment)), churn_by_segment.values, color='coral')
        ax8.set_yticks(range(len(churn_by_segment)))
        ax8.set_yticklabels(churn_by_segment.index)
        ax8.set_xlabel('Churn Rate (%)')
        ax8.set_title('Churn Rate by Segment', fontsize=14, fontweight='bold')
        
        # 9. RFM Heatmap
        ax9 = plt.subplot(3, 3, 9)
        rfm_corr = data[['recency', 'frequency', 'monetary', 'churn_probability']].corr()
        sns.heatmap(rfm_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax9)
        ax9.set_title('RFM Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'executive_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Executive dashboard saved to {FIGURES_DIR}")
    
    def generate_report(self, data, kpis, segment_insights, recommendations, top_priority):
        """Generate comprehensive business report"""
        print("\nGenerating Business Report...")
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        report_path = os.path.join(REPORTS_DIR, 'business_insights_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("CUSTOMER SEGMENTATION AND RETENTION ANALYSIS - EXECUTIVE REPORT\n")
            f.write("="*100 + "\n\n")
            
            f.write("1. EXECUTIVE SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"Total Customers: {kpis['total_customers']:,}\n")
            f.write(f"Retention Rate: {kpis['retention_rate']:.2f}%\n")
            f.write(f"Churn Rate: {kpis['churn_rate']:.2f}%\n")
            f.write(f"Total Revenue: ${kpis['total_revenue']:,.2f}\n")
            f.write(f"Average Customer Lifetime Value: ${kpis['avg_clv']:,.2f}\n")
            f.write(f"High Risk Customers: {kpis['high_risk_customers']:,}\n")
            f.write(f"Revenue at Risk: ${kpis['revenue_at_risk']:,.2f}\n\n")
            
            f.write("2. SEGMENT ANALYSIS\n")
            f.write("-" * 100 + "\n")
            f.write(segment_insights.to_string())
            f.write("\n\n")
            
            f.write("3. RETENTION RECOMMENDATIONS\n")
            f.write("-" * 100 + "\n")
            f.write(recommendations.to_string(index=False))
            f.write("\n\n")
            
            f.write("4. TOP PRIORITY CUSTOMERS (Top 20)\n")
            f.write("-" * 100 + "\n")
            f.write(top_priority.head(20).to_string(index=False))
            f.write("\n\n")
            
            f.write("5. KEY INSIGHTS\n")
            f.write("-" * 100 + "\n")
            f.write(f"• Top revenue segment: {segment_insights['total_revenue'].idxmax()}\n")
            f.write(f"• Highest churn risk segment: {segment_insights['churn_rate'].idxmax()}\n")
            f.write(f"• Most valuable segment (CLV): {segment_insights['avg_clv'].idxmax()}\n")
            f.write(f"• Immediate action required for {kpis['high_risk_customers']} high-risk customers\n")
            f.write(f"• Potential revenue recovery: ${kpis['revenue_at_risk']:,.2f}\n\n")
            
            f.write("="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        print(f"[OK] Business report saved to {report_path}")
        
        # Save data outputs
        segment_insights.to_csv(os.path.join(REPORTS_DIR, 'segment_analysis.csv'))
        recommendations.to_csv(os.path.join(REPORTS_DIR, 'retention_recommendations.csv'), index=False)
        top_priority.to_csv(os.path.join(REPORTS_DIR, 'top_priority_customers.csv'), index=False)

def main():
    insights_engine = BusinessInsights()
    
    # Load data
    data = insights_engine.load_data()
    
    # Calculate KPIs
    kpis = insights_engine.calculate_kpis(data)
    
    # Segment analysis
    segment_insights = insights_engine.segment_analysis(data)
    
    # Generate recommendations
    recommendations = insights_engine.generate_recommendations(data)
    
    # Customer prioritization
    top_priority = insights_engine.customer_prioritization(data)
    
    # Create executive dashboard
    insights_engine.create_executive_dashboard(data, kpis, segment_insights)
    
    # Generate report
    insights_engine.generate_report(data, kpis, segment_insights, recommendations, top_priority)
    
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS GENERATION COMPLETE")
    print("="*80)
    
    return insights_engine.insights

if __name__ == "__main__":
    insights = main()
