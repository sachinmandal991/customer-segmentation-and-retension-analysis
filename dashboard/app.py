import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_PROCESSED, REPORTS_DIR, RETENTION_STRATEGIES

# Page configuration
st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Compact header
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stMetric {background-color: white; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    h1 {font-size: 1.8rem !important; margin-top: 0 !important; padding-top: 0 !important;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all processed data"""
    customers = pd.read_csv(os.path.join(DATA_PROCESSED, 'customer_churn_predictions.csv'))
    segment_analysis = pd.read_csv(os.path.join(REPORTS_DIR, 'segment_analysis.csv'))
    recommendations = pd.read_csv(os.path.join(REPORTS_DIR, 'retention_recommendations.csv'))
    top_priority = pd.read_csv(os.path.join(REPORTS_DIR, 'top_priority_customers.csv'))
    return customers, segment_analysis, recommendations, top_priority

# Load data
try:
    customers, segment_analysis, recommendations, top_priority = load_data()
except:
    st.error("⚠️ Data not found. Please run the pipeline first: python src/data/data_loader.py && python src/data/preprocessing.py && python src/models/segmentation.py && python src/models/churn_prediction.py && python src/models/business_insights.py")
    st.stop()

# Sidebar
st.sidebar.title("📊 Menu")
page = st.sidebar.radio("", ["Overview", "Segmentation", "Churn Analysis", "Recommendations", "Customer Search"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
selected_segments = st.sidebar.multiselect(
    "Select Segments",
    options=customers['segment_name'].unique(),
    default=customers['segment_name'].unique()
)

selected_risk = st.sidebar.multiselect(
    "Churn Risk Level",
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

# Filter data
filtered_customers = customers[
    (customers['segment_name'].isin(selected_segments)) &
    (customers['churn_risk'].isin(selected_risk))
]

# OVERVIEW PAGE
if page == "Overview":
    st.title("🎯 Customer Analytics Dashboard")
    st.markdown("---")
    
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Customers", f"{len(customers):,}")
    with col2:
        retention_rate = (1 - customers['is_churned'].mean()) * 100
        st.metric("Retention Rate", f"{retention_rate:.1f}%", delta=f"{retention_rate-85:.1f}%")
    with col3:
        st.metric("Total Revenue", f"${customers['total_spent'].sum():,.0f}")
    with col4:
        st.metric("Avg CLV", f"${customers['clv'].mean():,.0f}")
    with col5:
        high_risk = len(customers[customers['churn_risk'] == 'High'])
        st.metric("High Risk", f"{high_risk:,}", delta=f"-{high_risk}", delta_color="inverse")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by Segment
        segment_revenue = customers.groupby('segment_name')['total_spent'].sum().reset_index()
        fig = px.pie(segment_revenue, values='total_spent', names='segment_name',
                    title='Revenue Distribution by Segment',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn Risk Distribution
        risk_counts = customers['churn_risk'].value_counts().reset_index()
        risk_counts.columns = ['churn_risk', 'count']
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        fig = px.bar(risk_counts, x='churn_risk', y='count',
                    title='Churn Risk Distribution',
                    color='churn_risk',
                    color_discrete_map=colors)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Count by Segment
        segment_counts = customers['segment_name'].value_counts().reset_index()
        segment_counts.columns = ['segment_name', 'count']
        fig = px.bar(segment_counts, x='count', y='segment_name',
                    title='Customers by Segment', orientation='h',
                    color='count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV Distribution
        fig = px.histogram(customers, x='clv', nbins=50,
                          title='Customer Lifetime Value Distribution',
                          labels={'clv': 'CLV ($)'})
        fig.add_vline(x=customers['clv'].mean(), line_dash="dash",
                     annotation_text=f"Mean: ${customers['clv'].mean():.0f}")
        st.plotly_chart(fig, use_container_width=True)

# SEGMENTATION PAGE
elif page == "Segmentation":
    st.title("👥 Customer Segmentation Analysis")
    
    # Segment Profile Table
    st.markdown("### Segment Profiles")
    st.dataframe(segment_analysis.style.background_gradient(cmap='YlOrRd', subset=['churn_rate']), 
                use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment Performance Matrix
        segment_perf = customers.groupby('segment_name').agg({
            'clv': 'mean',
            'churn_probability': 'mean',
            'customer_id': 'count'
        }).reset_index()
        
        fig = px.scatter(segment_perf, x='churn_probability', y='clv',
                        size='customer_id', color='segment_name',
                        title='Segment Performance Matrix',
                        labels={'churn_probability': 'Avg Churn Probability',
                               'clv': 'Avg CLV ($)',
                               'customer_id': 'Customer Count'},
                        hover_data=['segment_name'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RFM Analysis by Segment
        rfm_segment = customers.groupby('segment_name')[['recency', 'frequency', 'monetary']].mean().reset_index()
        rfm_melted = rfm_segment.melt(id_vars='segment_name', var_name='RFM Metric', value_name='Value')
        
        fig = px.bar(rfm_melted, x='segment_name', y='Value', color='RFM Metric',
                    title='RFM Profile by Segment', barmode='group')
        fig.update_xaxis(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Segment View
    st.markdown("### Detailed Segment Analysis")
    selected_segment = st.selectbox("Select Segment", customers['segment_name'].unique())
    
    segment_data = customers[customers['segment_name'] == selected_segment]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(segment_data):,}")
    with col2:
        st.metric("Avg CLV", f"${segment_data['clv'].mean():,.0f}")
    with col3:
        st.metric("Total Revenue", f"${segment_data['total_spent'].sum():,.0f}")
    with col4:
        st.metric("Churn Rate", f"{segment_data['is_churned'].mean()*100:.1f}%")

# CHURN ANALYSIS PAGE
elif page == "Churn Analysis":
    st.title("⚠️ Churn Prediction Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Churners", f"{customers['is_churned'].sum():,}")
    with col2:
        st.metric("Avg Churn Probability", f"{customers['churn_probability'].mean()*100:.1f}%")
    with col3:
        revenue_at_risk = customers[customers['churn_risk'] == 'High']['total_spent'].sum()
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn Probability Distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=customers[customers['is_churned'] == 0]['churn_probability'],
            name='Not Churned', opacity=0.7, marker_color='green'
        ))
        fig.add_trace(go.Histogram(
            x=customers[customers['is_churned'] == 1]['churn_probability'],
            name='Churned', opacity=0.7, marker_color='red'
        ))
        fig.update_layout(title='Churn Probability Distribution',
                         xaxis_title='Churn Probability',
                         yaxis_title='Count', barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn Rate by Segment
        churn_by_segment = customers.groupby('segment_name')['is_churned'].mean().reset_index()
        churn_by_segment['is_churned'] = churn_by_segment['is_churned'] * 100
        
        fig = px.bar(churn_by_segment, x='is_churned', y='segment_name',
                    title='Churn Rate by Segment (%)', orientation='h',
                    color='is_churned', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # High Risk Customers
    st.markdown("### High Risk Customers")
    high_risk_customers = customers[customers['churn_risk'] == 'High'][
        ['customer_id', 'segment_name', 'churn_probability', 'clv', 'total_spent', 'recency']
    ].sort_values('churn_probability', ascending=False).head(50)
    
    st.dataframe(high_risk_customers.style.background_gradient(cmap='Reds', subset=['churn_probability']),
                use_container_width=True)

# RECOMMENDATIONS PAGE
elif page == "Recommendations":
    st.title("💡 Retention Strategy Recommendations")
    
    st.markdown("### Segment-wise Recommendations")
    st.dataframe(recommendations, use_container_width=True)
    
    st.markdown("---")
    
    # Top Priority Customers
    st.markdown("### Top Priority Customers for Retention")
    st.dataframe(top_priority.head(20).style.background_gradient(cmap='RdYlGn_r', subset=['churn_probability']),
                use_container_width=True)
    
    st.markdown("---")
    
    # Action Plan
    st.markdown("### Recommended Action Plan")
    
    for idx, row in recommendations.iterrows():
        with st.expander(f"📌 {row['segment']} - Priority: {row['priority']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Customers:** {row['customer_count']:,}")
                st.write(f"**Revenue:** ${row['total_revenue']:,.0f}")
                st.write(f"**High Risk:** {row['high_risk_customers']:,}")
            with col2:
                st.write(f"**Avg Churn Probability:** {row['avg_churn_probability']*100:.1f}%")
                st.write(f"**Recommended Actions:**")
                st.write(row['recommended_actions'])

# CUSTOMER SEARCH PAGE
elif page == "Customer Search":
    st.title("🔍 Customer Search & Analysis")
    
    customer_id = st.text_input("Enter Customer ID", placeholder="e.g., CUST_00001")
    
    if customer_id:
        customer = customers[customers['customer_id'] == customer_id]
        
        if len(customer) > 0:
            customer = customer.iloc[0]
            
            st.markdown(f"## Customer Profile: {customer_id}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Segment", customer['segment_name'])
            with col2:
                st.metric("Churn Risk", customer['churn_risk'])
            with col3:
                st.metric("CLV", f"${customer['clv']:,.0f}")
            with col4:
                st.metric("Total Spent", f"${customer['total_spent']:,.0f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Customer Metrics")
                st.write(f"**Recency:** {customer['recency']} days")
                st.write(f"**Frequency:** {customer['frequency']} transactions")
                st.write(f"**Monetary:** ${customer['monetary']:,.2f}")
                st.write(f"**Churn Probability:** {customer['churn_probability']*100:.1f}%")
                st.write(f"**RFM Score:** {customer['rfm_score']}")
            
            with col2:
                st.markdown("### Recommendations")
                segment_name = customer['segment_name']
                
                # Map to strategy
                if 'High-Value' in segment_name:
                    strategies = RETENTION_STRATEGIES['high_value']
                elif 'Loyal' in segment_name:
                    strategies = RETENTION_STRATEGIES['loyal']
                elif 'At-Risk' in segment_name:
                    strategies = RETENTION_STRATEGIES['at_risk']
                elif 'Inactive' in segment_name:
                    strategies = RETENTION_STRATEGIES['inactive']
                elif 'New' in segment_name:
                    strategies = RETENTION_STRATEGIES['new']
                else:
                    strategies = RETENTION_STRATEGIES['potential']
                
                for strategy in strategies:
                    st.write(f"✓ {strategy}")
        else:
            st.warning("Customer not found!")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("📊 Customer Analytics Dashboard\n\nPowered by Streamlit & Python")
