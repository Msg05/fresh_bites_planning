import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="FreshBites Supply Chain Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FreshBitesOptimizer:
    def __init__(self):
        self.df = None
        self.plant_capacities = {'Delhi': 100, 'Pune': 80}
        self.PROFIT_MARGIN = 2.50
        self.HOLDING_COST = 0.20
        self.STOCK_OUT_COST = 4.00
        
    def generate_data(self):
        """Generate synthetic data"""
        np.random.seed(42)
        weeks = 35
        skus = ['Potato Chips', 'Nachos', 'Cookies', 'Energy Bar', 'Instant Noodles']
        regions = ['Mumbai', 'Kolkata', 'Delhi']
        
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
                    
                    data.append({
                        'Week_ID': week_id,
                        'Date': date,
                        'SKU': sku,
                        'Region': region,
                        'Forecast_Demand': round(forecast, 2),
                        'Actual_Demand': round(actual, 2),
                        'Current_Stock': current_stock,
                        'Is_Festival': is_festival,
                    })
        
        self.df = pd.DataFrame(data)
        self.df['Plant'] = self.df['Region'].apply(
            lambda x: 'Delhi' if x in ['Delhi', 'Kolkata'] else 'Pune'
        )
        
        # Create adjusted forecast
        self.df['Adjusted_Forecast'] = self.df.apply(self._create_adjusted_forecast, axis=1)
        
    def _create_adjusted_forecast(self, row):
        """Create adjusted forecast with business rules"""
        base_forecast = row['Forecast_Demand']
        
        if row['Is_Festival'] == 1:
            base_forecast *= 1.45
        
        if row['Region'] == 'Mumbai':
            base_forecast *= 1.10
        elif row['Region'] == 'Kolkata' and row['SKU'] != 'Cookies':
            base_forecast *= 0.80
        
        return max(5, base_forecast)
    
    def run_optimization(self, week_id):
        """Run production optimization using scipy.optimize"""
        week_data = self.df[self.df['Week_ID'] == week_id].copy()
        skus = week_data['SKU'].unique()
        plants = week_data['Plant'].unique()
        
        demand_dict = week_data.groupby('SKU')['Adjusted_Forecast'].first().to_dict()
        stock_dict = week_data.groupby('SKU')['Current_Stock'].first().to_dict()
        
        # Number of decision variables: (number of SKUs) * (number of plants)
        n_skus = len(skus)
        n_plants = len(plants)
        n_vars = n_skus * n_plants
        
        # Objective function: minimize stock-out (we'll use a simplified approach)
        # We want to minimize the cost of production and stock-out
        c = [0.001] * n_vars  # Small production cost
        
        # Constraints: Plant capacities
        A_ub = []
        b_ub = []
        
        for plant_idx, plant in enumerate(plants):
            # Create constraint row for this plant
            constraint_row = [0] * n_vars
            for sku_idx in range(n_skus):
                constraint_row[sku_idx * n_plants + plant_idx] = 1
            A_ub.append(constraint_row)
            b_ub.append(self.plant_capacities[plant])
        
        # Constraints: Demand fulfillment (simplified)
        # We'll use bounds instead of complex constraints for simplicity
        bounds = [(0, None)] * n_vars
        
        # Solve the linear programming problem
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                # Process results
                production_values = result.x
                results = []
                
                for sku_idx, sku in enumerate(skus):
                    for plant_idx, plant in enumerate(plants):
                        prod_value = production_values[sku_idx * n_plants + plant_idx]
                        if prod_value > 0.1:
                            results.append({
                                'SKU': sku,
                                'Plant': plant,
                                'Production_Allocated': round(prod_value, 2),
                                'Demand_Target': round(demand_dict[sku], 2),
                                'Current_Stock': stock_dict[sku]
                            })
                
                # Calculate estimated stock-out (simplified)
                total_demand = sum(demand_dict.values())
                total_stock = sum(stock_dict.values())
                total_production = sum(production_values)
                stock_out = max(0, total_demand - (total_stock + total_production))
                
                return pd.DataFrame(results), stock_out
            else:
                st.warning("Optimization didn't converge. Using heuristic approach.")
                return self._heuristic_optimization(week_data, demand_dict, stock_dict)
                
        except Exception as e:
            st.warning(f"Optimization failed: {e}. Using heuristic approach.")
            return self._heuristic_optimization(week_data, demand_dict, stock_dict)
    
    def _heuristic_optimization(self, week_data, demand_dict, stock_dict):
        """Fallback heuristic optimization"""
        results = []
        skus = week_data['SKU'].unique()
        plants = week_data['Plant'].unique()
        
        total_stock_out = 0
        
        for sku in skus:
            demand = demand_dict[sku]
            current_stock = stock_dict[sku]
            shortfall = max(0, demand - current_stock)
            
            if shortfall > 0:
                # Distribute production among plants based on capacity
                total_capacity = sum(self.plant_capacities.values())
                for plant in plants:
                    plant_share = self.plant_capacities[plant] / total_capacity
                    production = shortfall * plant_share
                    
                    if production > 0.1:
                        results.append({
                            'SKU': sku,
                            'Plant': plant,
                            'Production_Allocated': round(production, 2),
                            'Demand_Target': round(demand, 2),
                            'Current_Stock': current_stock
                        })
                
                total_stock_out += max(0, shortfall - sum([r['Production_Allocated'] for r in results if r['SKU'] == sku]))
        
        return pd.DataFrame(results), total_stock_out
    
    def calculate_performance(self, week_id, use_adjusted=True):
        """Calculate performance metrics for a week"""
        week_data = self.df[self.df['Week_ID'] == week_id]
        demand_target = 'Adjusted_Forecast' if use_adjusted else 'Forecast_Demand'
        
        total_demand = week_data[demand_target].sum()
        total_stock = week_data['Current_Stock'].sum()
        potential_sales = min(total_demand, total_stock)
        stock_out_units = max(0, total_demand - total_stock)
        excess_stock_units = max(0, total_stock - total_demand)
        
        revenue = potential_sales * self.PROFIT_MARGIN
        stock_out_cost = stock_out_units * self.STOCK_OUT_COST
        holding_cost = excess_stock_units * self.HOLDING_COST
        net_profit = revenue - stock_out_cost - holding_cost
        
        return {
            'total_demand': total_demand,
            'total_stock': total_stock,
            'potential_sales': potential_sales,
            'stock_out_units': stock_out_units,
            'excess_stock_units': excess_stock_units,
            'revenue': revenue,
            'stock_out_cost': stock_out_cost,
            'holding_cost': holding_cost,
            'net_profit': net_profit
        }
    
    def calculate_inventory_risk(self, week_id):
        """Calculate inventory risks for a week"""
        week_data = self.df[self.df['Week_ID'] == week_id].copy()
        
        week_data['safety_stock'] = week_data['Adjusted_Forecast'] * 0.5
        week_data['reorder_point'] = week_data['safety_stock'] * 1.5
        week_data['stock_out_risk'] = week_data['Current_Stock'] < week_data['safety_stock']
        week_data['overstock_risk'] = week_data['Current_Stock'] > (week_data['reorder_point'] * 2)
        
        return week_data

def main():
    st.markdown('<h1 class="main-header">📦 FreshBites Supply Chain Optimizer</h1>', unsafe_allow_html=True)
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = FreshBitesOptimizer()
        st.session_state.optimizer.generate_data()
    
    optimizer = st.session_state.optimizer
    
    # Sidebar
    st.sidebar.header("Configuration")
    selected_week = st.sidebar.selectbox(
        "Select Week",
        options=sorted(optimizer.df['Week_ID'].unique()),
        index=len(optimizer.df['Week_ID'].unique()) - 1
    )
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total SKUs", len(optimizer.df['SKU'].unique()))
    with col2:
        st.metric("Total Regions", len(optimizer.df['Region'].unique()))
    with col3:
        festival_weeks = optimizer.df['Is_Festival'].sum()
        st.metric("Festival Weeks", festival_weeks)
    
    # Week overview
    st.subheader(f"Week {selected_week} Overview")
    week_data = optimizer.df[optimizer.df['Week_ID'] == selected_week]
    is_festival = week_data['Is_Festival'].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Festival Week:** {'Yes' if is_festival else 'No'}")
    with col2:
        total_demand = week_data['Adjusted_Forecast'].sum()
        st.metric("Total Demand", f"{total_demand:.0f} units")
    with col3:
        total_stock = week_data['Current_Stock'].sum()
        st.metric("Current Stock", f"{total_stock:.0f} units")
    with col4:
        stock_ratio = (total_stock / total_demand * 100) if total_demand > 0 else 0
        st.metric("Stock Coverage", f"{stock_ratio:.1f}%")
    
    # Demand vs Stock chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=week_data['SKU'] + ' - ' + week_data['Region'],
        y=week_data['Adjusted_Forecast'],
        name='Adjusted Forecast',
        marker_color='#2E86AB'
    ))
    fig.add_trace(go.Bar(
        x=week_data['SKU'] + ' - ' + week_data['Region'],
        y=week_data['Current_Stock'],
        name='Current Stock',
        marker_color='#A23B72'
    ))
    fig.update_layout(
        title='Demand vs Current Stock by SKU-Region',
        barmode='group',
        height=400,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization section
    st.subheader("Production Optimization")
    if st.button("Run Optimization", type="primary"):
        with st.spinner("Running optimization..."):
            results_df, stock_out = optimizer.run_optimization(selected_week)
            
        if not results_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("✅ Optimization Complete!")
                st.metric("Projected Stock-Out", f"{stock_out:.1f} units")
                
                # Production summary
                st.write("**Production Summary:**")
                production_summary = results_df.groupby('Plant')['Production_Allocated'].sum()
                for plant, production in production_summary.items():
                    capacity = optimizer.plant_capacities[plant]
                    utilization = (production / capacity) * 100
                    st.write(f"- {plant}: {production:.0f} units ({utilization:.1f}% utilization)")
            
            with col2:
                # Production allocation chart
                fig = px.bar(
                    results_df, 
                    x='SKU', 
                    y='Production_Allocated', 
                    color='Plant',
                    title='Production Allocation by SKU and Plant',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.write("**Detailed Production Plan:**")
            st.dataframe(results_df.style.format({
                'Production_Allocated': '{:.1f}',
                'Demand_Target': '{:.1f}',
                'Current_Stock': '{:.0f}'
            }))
        else:
            st.info("No production allocation needed. Current stock is sufficient.")
    
    # Inventory Risk Analysis
    st.subheader("Inventory Risk Analysis")
    risk_data = optimizer.calculate_inventory_risk(selected_week)
    
    stock_out_risks = risk_data[risk_data['stock_out_risk']]
    overstock_risks = risk_data[risk_data['overstock_risk']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Stock-Out Risks:**")
        if not stock_out_risks.empty:
            for _, risk in stock_out_risks.iterrows():
                st.error(
                    f"🚨 {risk['SKU']} in {risk['Region']}: "
                    f"{risk['Current_Stock']} units (Safety: {risk['safety_stock']:.1f})"
                )
        else:
            st.success("No critical stock-out risks")
    
    with col2:
        st.write("**Overstock Risks:**")
        if not overstock_risks.empty:
            for _, risk in overstock_risks.iterrows():
                st.warning(
                    f"⚠️ {risk['SKU']} in {risk['Region']}: "
                    f"{risk['Current_Stock']} units (Max Optimal: {risk['reorder_point'] * 1.2:.1f})"
                )
        else:
            st.success("No overstock risks")
    
    # Performance comparison
    st.subheader("Performance Analysis")
    baseline_perf = optimizer.calculate_performance(selected_week, use_adjusted=False)
    optimized_perf = optimizer.calculate_performance(selected_week, use_adjusted=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Revenue Potential", f"₹{optimized_perf['revenue']:.0f}")
    with col2:
        st.metric("Stock-out Cost", f"₹{optimized_perf['stock_out_cost']:.0f}")
    with col3:
        st.metric("Holding Cost", f"₹{optimized_perf['holding_cost']:.0f}")
    with col4:
        st.metric("Net Profit", f"₹{optimized_perf['net_profit']:.0f}")
    
    # Data explorer
    st.sidebar.subheader("Data Explorer")
    if st.sidebar.checkbox("Show Raw Data"):
        st.sidebar.dataframe(optimizer.df)
    
    if st.sidebar.checkbox("Download Data"):
        csv = optimizer.df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="freshbites_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
