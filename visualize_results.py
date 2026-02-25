import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Maximize window
plt.rcParams['figure.max_open_warning'] = 50

# Load results
segments = pd.read_csv('data/processed/customer_segments.csv')
predictions = pd.read_csv('data/processed/customer_churn_predictions.csv')

# Set style
sns.set_style('whitegrid')

# Create comprehensive visualization - FULL SCREEN
fig = plt.figure(figsize=(24, 18))  # Larger size
fig.canvas.manager.full_screen_toggle()  # Full screen

# 1. Segment Distribution (Pie Chart)
ax1 = plt.subplot(3, 3, 1)
segment_counts = segments['segment_name'].value_counts()
ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')

# 2. Revenue by Segment (Bar Chart)
ax2 = plt.subplot(3, 3, 2)
revenue = segments.groupby('segment_name')['total_spent'].sum().sort_values(ascending=False)
ax2.barh(range(len(revenue)), revenue.values, color=['#2ecc71', '#3498db', '#e74c3c'])
ax2.set_yticks(range(len(revenue)))
ax2.set_yticklabels(revenue.index)
ax2.set_xlabel('Total Revenue ($)')
ax2.set_title('Revenue by Segment', fontsize=14, fontweight='bold')

# 3. Churn Risk Distribution (Bar Chart)
ax3 = plt.subplot(3, 3, 3)
risk_counts = predictions['churn_risk'].value_counts()
colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
bars = ax3.bar(risk_counts.index, risk_counts.values, color=[colors[x] for x in risk_counts.index])
ax3.set_ylabel('Number of Customers')
ax3.set_title('Churn Risk Distribution', fontsize=14, fontweight='bold')

# 4. CLV Distribution (Histogram)
ax4 = plt.subplot(3, 3, 4)
ax4.hist(predictions['clv'], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax4.axvline(predictions['clv'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${predictions["clv"].mean():.0f}')
ax4.set_xlabel('Customer Lifetime Value ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('CLV Distribution', fontsize=14, fontweight='bold')
ax4.legend()

# 5. Churn Rate by Segment (Bar Chart)
ax5 = plt.subplot(3, 3, 5)
churn_by_seg = predictions.groupby('segment_name')['is_churned'].mean().sort_values(ascending=False) * 100
ax5.barh(range(len(churn_by_seg)), churn_by_seg.values, color='coral')
ax5.set_yticks(range(len(churn_by_seg)))
ax5.set_yticklabels(churn_by_seg.index)
ax5.set_xlabel('Churn Rate (%)')
ax5.set_title('Churn Rate by Segment', fontsize=14, fontweight='bold')

# 6. RFM Scores (Box Plot)
ax6 = plt.subplot(3, 3, 6)
rfm_data = predictions[['recency', 'frequency', 'monetary']].copy()
rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
ax6.boxplot([rfm_data['Recency'], rfm_data['Frequency'], rfm_data['Monetary']], labels=['Recency', 'Frequency', 'Monetary'])
ax6.set_ylabel('Value')
ax6.set_title('RFM Metrics Distribution', fontsize=14, fontweight='bold')

# 7. Churn Probability Distribution (Histogram)
ax7 = plt.subplot(3, 3, 7)
ax7.hist(predictions[predictions['is_churned']==0]['churn_probability'], bins=30, alpha=0.6, label='Not Churned', color='green')
ax7.hist(predictions[predictions['is_churned']==1]['churn_probability'], bins=30, alpha=0.6, label='Churned', color='red')
ax7.set_xlabel('Churn Probability')
ax7.set_ylabel('Frequency')
ax7.set_title('Churn Probability Distribution', fontsize=14, fontweight='bold')
ax7.legend()

# 8. Segment Performance Matrix (Scatter)
ax8 = plt.subplot(3, 3, 8)
seg_perf = predictions.groupby('segment_name').agg({'clv': 'mean', 'churn_probability': 'mean', 'customer_id': 'count'})
scatter = ax8.scatter(seg_perf['churn_probability'], seg_perf['clv'], s=seg_perf['customer_id']/10, alpha=0.6, c=range(len(seg_perf)), cmap='viridis')
for idx, seg in enumerate(seg_perf.index):
    ax8.annotate(seg, (seg_perf['churn_probability'].iloc[idx], seg_perf['clv'].iloc[idx]), fontsize=9)
ax8.set_xlabel('Avg Churn Probability')
ax8.set_ylabel('Avg CLV ($)')
ax8.set_title('Segment Performance Matrix', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. Revenue at Risk (Stacked Bar)
ax9 = plt.subplot(3, 3, 9)
risk_revenue = predictions.groupby('churn_risk')['total_spent'].sum()
ax9.bar(risk_revenue.index, risk_revenue.values, color=[colors[x] for x in risk_revenue.index])
ax9.set_ylabel('Total Revenue ($)')
ax9.set_title('Revenue at Risk', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Comprehensive visualization saved!")
plt.get_current_fig_manager().window.state('zoomed')  # Maximize
plt.show()

# Additional: Segment Comparison
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Segment metrics comparison
seg_metrics = predictions.groupby('segment_name').agg({
    'customer_id': 'count',
    'total_spent': 'sum',
    'clv': 'mean',
    'is_churned': 'mean'
})

axes[0,0].bar(seg_metrics.index, seg_metrics['customer_id'], color='skyblue')
axes[0,0].set_title('Customer Count by Segment', fontsize=12, fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=45)

axes[0,1].bar(seg_metrics.index, seg_metrics['total_spent'], color='lightgreen')
axes[0,1].set_title('Total Revenue by Segment', fontsize=12, fontweight='bold')
axes[0,1].tick_params(axis='x', rotation=45)

axes[1,0].bar(seg_metrics.index, seg_metrics['clv'], color='orange')
axes[1,0].set_title('Average CLV by Segment', fontsize=12, fontweight='bold')
axes[1,0].tick_params(axis='x', rotation=45)

axes[1,1].bar(seg_metrics.index, seg_metrics['is_churned']*100, color='salmon')
axes[1,1].set_title('Churn Rate by Segment (%)', fontsize=12, fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/figures/segment_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Segment comparison saved!")
plt.get_current_fig_manager().window.state('zoomed')  # Maximize
plt.show()

print("\n[OK] All visualizations created successfully!")
print("Check outputs/figures/ folder for:")
print("  - comprehensive_analysis.png (9 charts)")
print("  - segment_comparison.png (4 charts)")
print("  - optimal_clusters.png (already generated)")
print("  - segment_analysis.png (already generated)")
print("  - churn_model_evaluation.png (already generated)")
print("  - executive_dashboard.png (already generated)")
