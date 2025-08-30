# freshbites_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="FreshBites Supply Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .section-header {font-size: 1.5rem; color: #2c3e50; margin: 1rem 0 0.5rem 0; padding-bottom: 0.3rem; border-bottom: 2px solid #1f77b4;}
    .metric-card {background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 8px; border-left: 4px solid #1f77b4;}
    .filter-section {background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;}
    .risk-critical {color: #ff4b4b; font-weight: bold;}
    .risk-warning {color: #ffa500; font-weight: bold;}
    .risk-good {color: #00cc96; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📦 FreshBites Supply Chain Optimizer</h1>', unsafe_allow_html=True)

# Sample data generation function
@st.cache_data
def load_data():
    np.random.seed(42)
    weeks = 35
    skus = ['Potato Chips', 'Nachos', 'Cookies', 'Energy Bar', 'Instant Noodles']
    regions = ['Mumbai', 'Kolkata', 'Delhi']
    suppliers = ['SUP_01', 'SUP_02', 'SUP_03']
    
    dates = pd.date_range(start='2023-01-01', periods=weeks, freq='W-SUN')
    data = []
    
    for week_id, date in enumerate(dates, 1):
        is_festival = 1 if week_id in [5, 10, 15, 20, 25, 30, 35] else 0
        
        for sku in skus:
            for region in regions:
                base_demand = np.random.randint(20, 40)
                
                if region == 'Mumbai':
                    base_demand *= 1.4
                elif region == 'Kolkata' and sku != 'Cookies':
                    base_demand *= 0.7

                forecast = base_demand + np.random.randint(-5, 5)
                actual = base_demand * (1 + is_festival * np.random.uniform(0.4, 0.5))
                actual += np.random.randint(-7, 7)
                actual = max(0, actual)
                
                if region == 'Mumbai':
                    current_stock = max(5, np.random.randint(0, 15))
                elif region == 'Kolkata' and sku == 'Cookies':
                    current_stock = np.random.randint(30, 50)
                else:
                    current_stock = np.random.randint(10, 25)
                
                # Calculate forecast error
                forecast_error = abs(actual - forecast) / forecast * 100 if forecast > 0 else 0
                
                data.append({
                    'Week_ID': week_id,
                    'Date': date,
                    'SKU': sku,
                    'Region': region,
                    'Forecast_Demand': round(forecast, 2),
                    'Actual_Demand': round(actual, 2),
                    'Current_Stock': current_stock,
                    'Is_Festival': is_festival,
                    'Forecast_Error_%': round(forecast_error, 1),
                    'Plant': 'Delhi' if region in ['Delhi', 'Kolkata'] else 'Pune',
                    'Supplier': np.random.choice(suppliers),
                    'OTIF_%': round(np.random.uniform(70, 95), 1)
                })
    
    df = pd.DataFrame(data)
    
    # Create adjusted forecast
    def create_adjusted_forecast(row):
        base_forecast = row['Forecast_Demand']
        if row['Is_Festival'] == 1:
            base_forecast *= 1.45
        if row['Region'] == 'Mumbai':
            base_forecast *= 1.10
        elif row['Region'] == 'Kolkata' and row['SKU'] != 'Cookies':
            base_forecast *= 0.80
        return max(5, round(base_forecast, 2))
    
    df['Adjusted_Forecast'] = df.apply(create_adjusted_forecast, axis=1)
    return df

# Load data
df = load_data()

# Sidebar with PROPER filters
st.sidebar.markdown('<div class="filter-section">', unsafe_allow_html=True)
st.sidebar.markdown('### 🔧 Filters & Controls')
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Week selector - FIXED
weeks = sorted(df['Week_ID'].unique())
selected_week = st.sidebar.selectbox("**Select Week**", weeks, index=0)

# Region selector - FIXED (proper names)
regions = ['Mumbai', 'Kolkata', 'Delhi']
selected_region = st.sidebar.selectbox("**Select Location**", regions, index=0)

# SKU multi-select - FIXED (proper names)
skus = ['Potato Chips', 'Nachos', 'Cookies', 'Energy Bar', 'Instant Noodles']
selected_skus = st.sidebar.multiselect("**Select Product**", skus, default=skus)

# Additional filters
show_only_festival = st.sidebar.checkbox("🎪 Show only Festival Weeks")
show_supplier_data = st.sidebar.checkbox("🏭 Show Supplier Analytics")

# Filter data based on selections
filtered_df = df[
    (df['Week_ID'] == selected_week) & 
    (df['Region'] == selected_region) & 
    (df['SKU'].isin(selected_skus))
]

if show_only_festival:
    filtered_df = filtered_df[filtered_df['Is_Festival'] == 1]

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Adj. Forecast", f"{filtered_df['Adjusted_Forecast'].sum():.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Actual Demand", f"{filtered_df['Actual_Demand'].sum():.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    stock_out_risk = (filtered_df['Current_Stock'] < filtered_df['Adjusted_Forecast'] * 0.5).sum()
    st.metric("Stockout Risks", f"{stock_out_risk}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    production_needed = max(0, (filtered_df['Adjusted_Forecast'] - filtered_df['Current_Stock']).sum())
    st.metric("POS Needed", f"{production_needed:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast Analysis", "🏭 Production Plan", "📦 Inventory Health", "🏭 Supplier Reliability"])

with tab1:
    st.markdown('<h3 class="section-header">Forecast vs Actual Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Forecast vs Actual by Product
        forecast_data = filtered_df.groupby('SKU')[['Adjusted_Forecast', 'Actual_Demand']].sum().reset_index()
        fig = px.bar(forecast_data, x='SKU', y=['Adjusted_Forecast', 'Actual_Demand'],
                     title='Forecast vs Actual by Product', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Forecast Error by SKU
        error_data = filtered_df.groupby('SKU')['Forecast_Error_%'].mean().reset_index()
        fig = px.bar(error_data, x='SKU', y='Forecast_Error_%',
                     title='Average Forecast Error by SKU (%)',
                     color='Forecast_Error_%', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # APE Table (Absolute Percentage Error)
    st.markdown('<h4 class="section-header">Forecast Accuracy (APE %) by SKU-Location</h4>', unsafe_allow_html=True)
    ape_df = filtered_df[['Week_ID', 'SKU', 'Region', 'Forecast_Error_%']].copy()
    ape_df = ape_df.rename(columns={'Forecast_Error_%': 'APE_%'})
    st.dataframe(ape_df.style.format({'APE_%': '{:.1f}'}).highlight_max(axis=0, color='#ffcccc').highlight_min(axis=0, color='#ccffcc'))

with tab2:
    st.markdown('<h3 class="section-header">Production Planning</h3>', unsafe_allow_html=True)
    
    # Production planning logic
    plant_capacities = {'Delhi': 100, 'Pune': 80}
    
    production_needs = []
    for _, row in filtered_df.iterrows():
        need = max(0, row['Adjusted_Forecast'] - row['Current_Stock'])
        if need > 0:
            # Allocation based on plant capacity
            delhi_alloc = need * (plant_capacities['Delhi'] / sum(plant_capacities.values()))
            pune_alloc = need - delhi_alloc
            
            production_needs.append({
                'Product': row['SKU'],
                'Production_Needed': need,
                'Delhi_Allocation': delhi_alloc,
                'Pune_Allocation': pune_alloc,
                'Current_Stock': row['Current_Stock'],
                'Demand': row['Adjusted_Forecast']
            })
    
    if production_needs:
        production_df = pd.DataFrame(production_needs)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Production Allocation Plan**")
            display_df = production_df.copy()
            display_df = display_df.round(1)
            st.dataframe(display_df.style.format({
                'Production_Needed': '{:.1f}',
                'Delhi_Allocation': '{:.1f}',
                'Pune_Allocation': '{:.1f}'
            }))
        
        with col2:
            # Capacity utilization
            delhi_util = (production_df['Delhi_Allocation'].sum() / plant_capacities['Delhi']) * 100
            pune_util = (production_df['Pune_Allocation'].sum() / plant_capacities['Pune']) * 100
            avg_util = (delhi_util + pune_util) / 2
            
            st.metric("Avg. Utilization %", f"{avg_util:.1f}%")
            
            util_data = pd.DataFrame({
                'Plant': ['Delhi', 'Pune'],
                'Utilization (%)': [delhi_util, pune_util]
            })
            
            fig = px.bar(util_data, x='Plant', y='Utilization (%)',
                         title='Plant Capacity Utilization',
                         color='Utilization (%)', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No production needed - current stock is sufficient!")

with tab3:
    st.markdown('<h3 class="section-header">Inventory Health Dashboard</h3>', unsafe_allow_html=True)
    
    inventory_health = filtered_df.copy()
    inventory_health['Status'] = np.where(
        inventory_health['Current_Stock'] < inventory_health['Adjusted_Forecast'] * 0.5,
        '🟥 Critical Risk',
        np.where(
            inventory_health['Current_Stock'] > inventory_health['Adjusted_Forecast'] * 1.5,
            '🟨 Overstock',
            '🟩 Optimal'
        )
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Inventory status pie chart
        status_count = inventory_health['Status'].value_counts()
        fig = px.pie(values=status_count.values, names=status_count.index,
                     title='Inventory Health Status', color=status_count.index,
                     color_discrete_map={'🟥 Critical Risk': 'red', '🟨 Overstock': 'orange', '🟩 Optimal': 'green'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Inventory Details**")
        for status in ['🟥 Critical Risk', '🟨 Overstock', '🟩 Optimal']:
            items = inventory_health[inventory_health['Status'] == status]
            if not items.empty:
                st.subheader(f"{status} ({len(items)} items)")
                for _, item in items.iterrows():
                    st.write(f"- {item['SKU']}: {item['Current_Stock']} units (Demand: {item['Adjusted_Forecast']:.1f})")

with tab4:
    st.markdown('<h3 class="section-header">Supplier Reliability Analytics</h3>', unsafe_allow_html=True)
    
    if show_supplier_data:
        supplier_data = df.groupby('Supplier').agg({
            'OTIF_%': 'mean',
            'SKU': 'count'
        }).reset_index()
        supplier_data = supplier_data.rename(columns={'SKU': 'Order_Count'})
        supplier_data['OTIF_%'] = supplier_data['OTIF_%'].round(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Supplier OTIF Performance**")
            fig = px.bar(supplier_data, x='Supplier', y='OTIF_%',
                         title='Supplier Reliability (Avg OTIF %)',
                         color='OTIF_%', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Purchase Order Actions**")
            po_data = pd.DataFrame({
                'Supplier': ['SUP_01', 'SUP_02', 'SUP_03'],
                'Place_PO': ['Yes', 'No', 'Yes'],
                'Reorder_Quantity': [376.7, 0, 268.1]
            })
            st.dataframe(po_data.style.format({'Reorder_Quantity': '{:.1f}'}))
        
        st.write("**Collaboration Summary**")
        st.info("""
        - **SUP_01**: High reliability (92.3% OTIF), recommend increase orders
        - **SUP_02**: Poor performance (76.1% OTIF), place on watchlist  
        - **SUP_03**: Good performance (88.7% OTIF), maintain current levels
        """)
    else:
        st.warning("Enable 'Show Supplier Analytics' in filters to view this data")

# Footer
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="📥 Download Current View",
    data=filtered_df.to_csv(index=False),
    file_name=f"freshbites_week_{selected_week}_{selected_region}.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**FreshBites Supply Optimizer**  
📊 AI-powered supply chain management  
🚀 Built for Hackathon 2024
""")
