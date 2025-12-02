import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
import os

# --- Configuration Constants ---
SIM_RUNS = 1000
CURRENT_AGENT_COUNT = 105
# Removed FILE_NAME and USER_NAME constants as data is in the same directory

# NEW CONSTANT: Data file name
DATA_FILE_NAME = 'pnldataset.csv'

# --- Model Definitions ---
MODEL_OPTIONS = {
    'Linear (1st Degree)': LinearRegression,
    'Quadratic (2nd Degree)': 'Poly_2',
    'Cubic (3rd Degree)': 'Poly_3',
    'Logarithmic (Log10)': 'Log10',
    'Natural Logarithmic (Ln)': 'Ln'
}

# --- Plotting Constants ---
AGGREGATE_COLORS = {
    'Income': 'blue',
    'Payroll': 'purple',
    'Marketing': 'red',
    'Overhead': 'orange'
}

# =================================================================
# GLOBAL PREDICTOR CLASSES (Required for caching)
# =================================================================

class PolynomialPredictor:
    """Helper class for polynomial feature transformation and prediction."""
    def __init__(self, model, poly_features):
        self.model = model
        self.poly_features = poly_features

    def predict(self, X):
        X_reshaped = np.array(X).reshape(-1, 1)
        X_poly = self.poly_features.transform(X_reshaped)
        return self.model.predict(X_poly).flatten()

class SKLearnPredictor:
    """Helper class for prediction with standard SKLearn models."""
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X_reshaped = np.array(X).reshape(-1, 1)
        return self.model.predict(X_reshaped).flatten()

class LogPredictor:
    """Helper class for logarithmic transformation and prediction."""
    def __init__(self, model, log_type='log10'):
        self.model = model
        self.log_type = log_type

    def predict(self, X):
        X_reshaped = np.array(X).reshape(-1, 1)
        # Add small epsilon to avoid log(0)
        X_safe = np.maximum(X_reshaped, 1e-10)
        if self.log_type == 'log10':
            X_transformed = np.log10(X_safe)
        else:  # natural log
            X_transformed = np.log(X_safe)
        return self.model.predict(X_transformed).flatten()

# =================================================================
# END GLOBAL CLASSES
# =================================================================

@st.cache_data
# MODIFICATION: Changed to use DATA_FILE_NAME constant
def load_data_and_calculate_regressions(file_path):
    """Loads data and pre-calculates regression models for all P&L items."""
    try:
        # MODIFICATION: Use pd.read_csv instead of pd.read_excel
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found at: {file_path}. Please ensure '{DATA_FILE_NAME}' is in the same directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

    critical_columns = [
        'Income', 'Marketing', 'Payroll', 'Overhead', 'Agent Count', 'AgentsPerforming'
    ]
    all_potential_columns = [
        'Income', 'Advances', 'Chargebacks', 'OtherIncome', 'Marketing', 
        'Payroll', 'PayrollSales', 'PayrollAdmin', 'PayrollOther', 'Overhead', 
        'Agent Count', 'AgentsPerforming'
    ]

    for col in all_potential_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=critical_columns, inplace=True)
    
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str)

    if df.empty:
        st.error("Data Error: The historical data is empty after removing rows with missing values in critical columns.")
        return None, None, None, None

    reg_data = {}
    x_column = 'Agent Count' 

    regression_targets = [col for col in all_potential_columns if col not in ['Agent Count', 'AgentsPerforming']]
    
    for col in regression_targets:
        reg_data[col] = {}
        
        for driver_col in ['Agent Count', 'AgentsPerforming']:
            
            reg_data[col][driver_col] = {}
            
            # Dynamic Data Filtering for the specific (col, driver_col) pair
            temp_df = df[[driver_col, col]].copy().dropna()
            
            if temp_df.shape[0] < 2:
                reg_data[col][driver_col] = {
                    name: {'predictor': lambda x: np.zeros_like(x).flatten(), 'r_squared': 0.0, 'sigma': 1.0, 'model_name': name, 'driver': driver_col}
                    for name in MODEL_OPTIONS.keys()
                }
                continue

            X_fit = temp_df[[driver_col]]
            Y_fit = temp_df[col].values.reshape(-1, 1)

            for name, model_class in MODEL_OPTIONS.items():
                
                is_poly = 'Degree' in name
                is_log = 'Log' in name or 'Ln' in name
                
                if is_poly:
                    degree = int(name.split('(')[-1].split('st')[0].split('nd')[0].split('rd')[0].strip())
                    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                    X_poly = poly_features.fit_transform(X_fit)
                    model = LinearRegression()
                    model.fit(X_poly, Y_fit)
                    y_pred = model.predict(X_poly)
                    predictor = PolynomialPredictor(model, poly_features).predict

                elif is_log:
                    # Add small epsilon to avoid log(0)
                    X_safe = np.maximum(X_fit.values, 1e-10)
                    if 'Log10' in name:
                        X_transformed = np.log10(X_safe)
                        log_type = 'log10'
                    else:  # Natural log
                        X_transformed = np.log(X_safe)
                        log_type = 'ln'
                    
                    model = LinearRegression()
                    model.fit(X_transformed, Y_fit)
                    y_pred = model.predict(X_transformed)
                    predictor = LogPredictor(model, log_type).predict

                else:
                    model = model_class()
                    model.fit(X_fit, Y_fit)
                    y_pred = model.predict(X_fit)
                    predictor = SKLearnPredictor(model).predict

                r_squared = r2_score(Y_fit, y_pred)
                mse = mean_squared_error(Y_fit, y_pred)
                rmse = np.sqrt(mse)
                sigma = np.std(Y_fit.ravel() - y_pred.ravel())
                
                reg_data[col][driver_col][name] = {
                    'predictor': predictor,
                    'r_squared': r_squared,
                    'mse': mse,
                    'rmse': rmse,
                    'sigma': sigma,
                    'model_name': name,
                    'driver': driver_col
                }

    return df, reg_data, ['Income', 'Marketing', 'Payroll', 'Overhead'], x_column 

def define_scenarios(base_agents, sim_months):
    """Generates the Agent Count and AgentsPerforming arrays for the simulation period."""
    
    last_performing_ratio = CURRENT_AGENT_COUNT / (CURRENT_AGENT_COUNT if CURRENT_AGENT_COUNT > 0 else 1)
    
    s1_agents = np.full(sim_months, base_agents)
    s2_agents = np.maximum(base_agents - np.arange(1, sim_months + 1) * 3, 10) 
    
    s1_performing = np.minimum((s1_agents * last_performing_ratio), s1_agents).astype(int)
    s2_performing = np.minimum((s2_agents * last_performing_ratio), s2_agents).astype(int)
    
    # --- Scenario 3: Hiring with 1-Month Lag ---
    s3_agents = base_agents + np.arange(1, sim_months + 1) * 3
    
    s3_performing = np.zeros_like(s3_agents)
    s3_performing[0] = np.minimum(CURRENT_AGENT_COUNT * last_performing_ratio, CURRENT_AGENT_COUNT).astype(int)
    
    for i in range(1, sim_months):
        s3_performing[i] = np.minimum(s3_agents[i-1] * last_performing_ratio, s3_agents[i-1]).astype(int)
        
    s3_performing = np.minimum(s3_performing, s3_agents)

    return {
        "Scenario 1: Maintain Headcount (105)": {'Agent Count': s1_agents, 'AgentsPerforming': s1_performing},
        "Scenario 2: Lay off 3/mo": {'Agent Count': s2_agents, 'AgentsPerforming': s2_performing},
        "Scenario 3: Add 3/mo": {'Agent Count': s3_agents, 'AgentsPerforming': s3_performing}
    }

def run_monte_carlo_scenarios(reg_data, selected_fits, scenario_agent_counts, sim_months, analysis_mode):
    """Runs the Monte Carlo simulation based on the selected analysis mode."""
    
    scenario_results = {}
    
    with st.spinner(f"Running {SIM_RUNS} MC simulations using {analysis_mode} model over {sim_months} months..."):
        for name, drivers in scenario_agent_counts.items():
            
            results = {}
            AC = drivers['Agent Count']
            AP = drivers['AgentsPerforming']
            
            # Helper function to run prediction and add variance
            def predict_and_simulate(col, driver_key, driver_data):
                reg_type = selected_fits.get(col, 'Linear (1st Degree)') 
                driver_lookup = 'Agent Count' if driver_key == 'Agent Count' else 'AgentsPerforming'
                
                model = reg_data[col][driver_lookup].get(reg_type, reg_data[col][driver_lookup]['Linear (1st Degree)'])
                
                base_predictions = model['predictor'](driver_data)
                random_errors = norm.rvs(loc=0, scale=model['sigma'], size=(SIM_RUNS, sim_months))
                return base_predictions + random_errors

            if analysis_mode == 'Basic (Univariate)':
                # BASIC: Uses Agent Count (AC) for ALL 4 primary lines
                for col in ['Income', 'Marketing', 'Payroll', 'Overhead']:
                    results[col] = predict_and_simulate(col, 'Agent Count', AC) 

            elif analysis_mode == 'Advanced (Composite)':
                
                # ADVANCED: Uses specific drivers for components
                component_map = {
                    'Advances': 'AgentsPerforming', 'Chargebacks': 'AgentsPerforming', 'OtherIncome': 'AgentsPerforming',
                    'Marketing': 'AgentsPerforming',
                    'PayrollSales': 'Agent Count', 'PayrollAdmin': 'Agent Count', 'PayrollOther': 'Agent Count',
                    'Overhead': 'Agent Count'
                }
                
                for col, driver_key in component_map.items():
                    driver_data = AP if driver_key == 'AgentsPerforming' else AC
                    results[col] = predict_and_simulate(col, driver_key, driver_data)

                # --- Composite Summation ---
                results['Income'] = results['Advances'] + results['Chargebacks'] + results['OtherIncome']
                results['Payroll'] = results['PayrollSales'] + results['PayrollAdmin'] + results['PayrollOther']
                
            # --- Final Profit Calculation ---
            total_expense = results['Marketing'] + results['Payroll'] + results['Overhead']
            results['Master Profit'] = results['Income'] - total_expense
            
            scenario_results[name] = results
            
    return scenario_results

def plot_historical_fit(df, col, reg_data, selected_fit_name, driver_col):
    """Generates an interactive Plotly figure for historical fit."""
    
    reg_details = reg_data[col][driver_col][selected_fit_name]
    predictor = reg_details['predictor']
    
    temp_df = df[[driver_col, col]].copy().dropna()
    if temp_df.empty:
        return go.Figure().update_layout(title="Not enough clean data to plot regression.")
        
    X_vals = temp_df[driver_col].values
    
    custom_data_columns = [col, driver_col]
    hover_data_template = f'{col}: %{{y:,.0f}}<br>{driver_col}: %{{x:,.0f}}'

    if 'Month' in df.columns:
        custom_data_columns.append('Month')
        hover_data_template += '<br>Month: %{customdata[2]}'
        df_plot_scatter = df.loc[temp_df.index]
    else:
        df_plot_scatter = temp_df 

    fig = px.scatter(
        df_plot_scatter, 
        x=driver_col, 
        y=col, 
        title=f"{col} Historical Data vs. {driver_col}",
        opacity=0.6,
        color_discrete_sequence=['blue'],
        custom_data=custom_data_columns
    )
    
    x_fit = np.linspace(X_vals.min(), X_vals.max(), 100)
    y_fit = predictor(x_fit)
    
    fig.add_trace(go.Scatter(
        x=x_fit, 
        y=y_fit, 
        mode='lines', 
        name=f"{reg_details['model_name']} Fit", 
        line=dict(color='red', dash='dash'),
        hovertemplate=f'Fit: %{{y:,.0f}}<br>{driver_col}: %{{x:,.0f}}<extra></extra>' 
    ))

    fig.update_traces(selector=dict(mode='markers'), hovertemplate=hover_data_template + '<extra></extra>')
    
    fig.update_layout(
        xaxis_title=driver_col,
        yaxis_title="Value ($)",
        height=450,
        margin={'t': 50, 'b': 20},
        showlegend=True
    )
    
    return fig

def plot_agent_trajectory(tab, agent_counts, sim_months, title):
    """Plots the monthly agent count using Plotly for interactivity."""
    months = np.arange(1, sim_months + 1)
    
    agent_df = pd.DataFrame({'Month': months, 'Total Agent Count': agent_counts['Agent Count'], 'Agents Performing': agent_counts['AgentsPerforming']})
    
    df_melt = agent_df.melt(id_vars=['Month'], var_name='Metric', value_name='Count')
    
    fig = px.line(
        df_melt, 
        x='Month', 
        y='Count', 
        color='Metric',
        title=f"Agent Count Trajectory: {title}",
        markers=True,
        color_discrete_sequence=['green', 'orange']
    )
    fig.update_layout(height=300, margin={'l': 20, 'r': 20, 't': 40, 'b': 20})
    fig.update_traces(hovertemplate='Month: %{x}<br>%{customdata[0]}: %{y:.0f}<extra></extra>', customdata=np.stack((df_melt['Metric']), axis=-1))
    
    tab.plotly_chart(fig, use_container_width=True, key=f"agent_trajectory_{title}")
    tab.info(f"Total Agents: Start at **{agent_counts['Agent Count'][0]:.0f}**, End at **{agent_counts['Agent Count'][-1]:.0f}**")


def plot_aggregate_pdf(tab, scenario_results, scenario_name, sim_months):
    """Plots the aggregate profit distribution using Plotly for interactivity, with percentile hover."""
    
    monthly_profit = scenario_results[scenario_name]['Master Profit']
    total_profit_runs = monthly_profit.sum(axis=1)
    
    mean_profit = np.mean(total_profit_runs)
    p5, p25, p50, p75, p95 = np.percentile(total_profit_runs, [5, 25, 50, 75, 95])
    
    fig = go.Figure(data=[go.Histogram(
        x=total_profit_runs, 
        nbinsx=30, 
        name='Total Profit', 
        marker_color='skyblue',
        hovertemplate='Profit Range: %{x}<br>Frequency: %{y}<extra></extra>'
    )])
    
    # Add vertical lines for key statistics
    fig.add_vline(x=mean_profit, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: ${mean_profit:,.0f}", annotation_position="top left", name="Mean")
    
    fig.add_vline(x=p50, line_dash="solid", line_color="red", line_width=2,
                  annotation_text=f"Median: ${p50:,.0f}", annotation_position="top", name="Median")
    fig.add_vline(x=p5, line_dash="dash", line_color="green", 
                  annotation_text=f"5th %ile: ${p5:,.0f}", annotation_position="top right", name="5th %ile")
    fig.add_vline(x=p95, line_dash="dash", line_color="green", 
                  annotation_text=f"95th %ile: ${p95:,.0f}", annotation_position="top right", name="95th %ile")
                    
    fig.add_vline(x=p25, line_dash="dot", line_color="gray", 
                  annotation_text=f"25th %ile", annotation_position="top", name="25th %ile")
    fig.add_vline(x=p75, line_dash="dot", line_color="gray", 
                  annotation_text=f"75th %ile", annotation_position="top", name="75th %ile")

    fig.update_layout(
        title=f"{sim_months}-Month Total Master Profit Distribution ({SIM_RUNS} Runs)",
        xaxis_title="Total Profit ($)",
        yaxis_title="Count / Frequency",
        height=450,
        margin={'t': 50, 'b': 20}
    )
    
    tab.plotly_chart(fig, use_container_width=True, key=f"profit_dist_{scenario_name}")

    # Display summary metrics
    col1, col2, col3, col4 = tab.columns(4)
    
    col1.metric("Mean Total Profit", f"${np.mean(total_profit_runs):,.0f}")
    col2.metric("Worst Case (5th %ile)", f"${p5:,.0f}")
    col3.metric("Best Case (95th %ile)", f"${p95:,.0f}")
    col4.markdown(f"**90% CI Range:**\n${p5:,.0f} to ${p95:,.0f}")
    
    tab.markdown("---")

def plot_breakeven_distribution(tab, scenario_results, scenario_name, sim_months):
    """Plots the distribution of months until breakeven across all Monte Carlo runs."""
    
    monthly_profit = scenario_results[scenario_name]['Master Profit']
    cumulative_profit = np.cumsum(monthly_profit, axis=1)
    
    # Find the first month where cumulative profit > 0 AND stays positive for the rest of the simulation
    breakeven_months = np.full(SIM_RUNS, sim_months + 1)  # Default to "never" (sim_months + 1)
    
    for run_idx in range(SIM_RUNS):
        for month_idx in range(sim_months):
            # Check if this month is positive AND all subsequent months stay positive
            if cumulative_profit[run_idx, month_idx] > 0:
                # Check if cumulative profit stays positive for all remaining months
                if np.all(cumulative_profit[run_idx, month_idx:] > 0):
                    breakeven_months[run_idx] = month_idx + 1  # +1 because months are 1-indexed
                    break  # Found the breakeven point, move to next run
    
    # Count runs that never break even
    never_breakeven = np.sum(breakeven_months > sim_months)
    breakeven_achieved = np.sum(breakeven_months <= sim_months)
    
    fig = go.Figure()
    
    # Add histogram for months that do break even
    if breakeven_achieved > 0:
        breakeven_data = breakeven_months[breakeven_months <= sim_months]
        fig.add_trace(go.Histogram(
            x=breakeven_data,
            nbinsx=sim_months,
            name='Breakeven Achieved',
            marker_color='lightgreen',
            hovertemplate='Month: %{x}<br>Frequency: %{y}<extra></extra>',
            xbins=dict(start=0.5, end=sim_months + 0.5, size=1)
        ))
    
    # Add bar for "never breakeven" category at position sim_months + 1
    if never_breakeven > 0:
        fig.add_trace(go.Bar(
            x=[sim_months + 1],
            y=[never_breakeven],
            name='Never Breaks Even',
            marker_color='lightcoral',
            hovertemplate=f'{sim_months + 1}+ Months: %{{y}}<extra></extra>',
            width=0.8,
            text=[f'{sim_months + 1}+'],
            textposition='outside'
        ))
    
    # Calculate statistics for runs that do break even
    if breakeven_achieved > 0:
        breakeven_data = breakeven_months[breakeven_months <= sim_months]
        mean_breakeven = np.mean(breakeven_data)
        median_breakeven = np.median(breakeven_data)
        
        # Add vertical lines for statistics
        fig.add_vline(x=median_breakeven, line_dash="solid", line_color="red", line_width=2,
                      annotation_text=f"Median: {median_breakeven:.1f} mo", 
                      annotation_position="top")
        fig.add_vline(x=mean_breakeven, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_breakeven:.1f} mo", 
                      annotation_position="top left")
    
    # Update x-axis to include the "never breakeven" bucket
    fig.update_layout(
        title=f"Months Until Breakeven Distribution ({SIM_RUNS} Runs)",
        xaxis_title="Months Until Cumulative Profit > $0",
        yaxis_title="Count / Frequency",
        height=450,
        margin={'t': 50, 'b': 20},
        showlegend=True,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, sim_months + 1)) + [sim_months + 1],
            ticktext=list(range(1, sim_months + 1)) + [f'{sim_months + 1}+']
        )
    )
    
    tab.plotly_chart(fig, use_container_width=True, key=f"breakeven_dist_{scenario_name}")
    
    # Display summary metrics
    col1, col2, col3, col4 = tab.columns(4)
    
    pct_breakeven = (breakeven_achieved / SIM_RUNS) * 100
    col1.metric("Runs Breaking Even", f"{breakeven_achieved} ({pct_breakeven:.1f}%)")
    col2.metric("Runs Never Breaking Even", f"{never_breakeven} ({100-pct_breakeven:.1f}%)")
    
    if breakeven_achieved > 0:
        col3.metric("Median Months to Breakeven", f"{median_breakeven:.1f}")
        col4.metric("Mean Months to Breakeven", f"{mean_breakeven:.1f}")
    else:
        col3.metric("Median Months to Breakeven", "N/A")
        col4.metric("Mean Months to Breakeven", "N/A")
    
    tab.markdown("---")
    
    # --- Investment to Breakeven Distribution ---
    plot_investment_to_breakeven(tab, scenario_results, scenario_name, sim_months, breakeven_months)

def plot_investment_to_breakeven(tab, scenario_results, scenario_name, sim_months, breakeven_months):
    """Plots the distribution of cumulative investment needed until breakeven (stacked by breakeven status)."""
    
    monthly_profit = scenario_results[scenario_name]['Master Profit']
    cumulative_profit = np.cumsum(monthly_profit, axis=1)
    
    # Separate runs that break even from those that don't
    breaks_even_mask = breakeven_months <= sim_months
    
    # Calculate the investment needed (cumulative negative profit) until breakeven
    investment_breakeven = []
    max_capital_drain_never = []
    
    for run_idx in range(SIM_RUNS):
        breakeven_month = breakeven_months[run_idx]
        
        if breakeven_month <= sim_months:
            # Find the minimum cumulative profit (most negative point) before breakeven
            min_cumulative = np.min(cumulative_profit[run_idx, :breakeven_month])
            # Cap at zero - negative means no additional investment needed
            investment_breakeven.append(max(0, -min_cumulative))
        else:
            # For runs that never break even, track the maximum capital drain
            min_cumulative = np.min(cumulative_profit[run_idx, :])
            max_capital_drain_never.append(-min_cumulative)
    
    investment_breakeven = np.array(investment_breakeven)
    max_capital_drain_never = np.array(max_capital_drain_never)
    
    fig = go.Figure()
    
    # Add histogram for runs that break even (light orange)
    if len(investment_breakeven) > 0:
        fig.add_trace(go.Histogram(
            x=investment_breakeven,
            nbinsx=30,
            name='Breaks Even',
            marker_color='#FFB366',  # Light orange
            hovertemplate='Investment to Breakeven: $%{x:,.0f}<br>Frequency: %{y}<extra></extra>'
        ))
    
    # Add histogram for runs that never break even (red)
    if len(max_capital_drain_never) > 0:
        fig.add_trace(go.Histogram(
            x=max_capital_drain_never,
            nbinsx=30,
            name='Never Breaks Even',
            marker_color='#FF6B6B',  # Red
            hovertemplate='Max Capital Drain: $%{x:,.0f}<br>Frequency: %{y}<extra></extra>'
        ))
    
    # Stack the histograms
    fig.update_layout(barmode='stack')
    
    # Calculate statistics ONLY for runs that break even
    if len(investment_breakeven) > 0:
        mean_investment = np.mean(investment_breakeven)
        median_investment = np.median(investment_breakeven)
        p5_investment = np.percentile(investment_breakeven, 5)
        p95_investment = np.percentile(investment_breakeven, 95)
        
        # Add vertical lines for statistics
        fig.add_vline(x=median_investment, line_dash="solid", line_color="darkred", line_width=2,
                      annotation_text=f"Median: ${median_investment:,.0f}", 
                      annotation_position="top")
        fig.add_vline(x=mean_investment, line_dash="dash", line_color="darkred",
                      annotation_text=f"Mean: ${mean_investment:,.0f}", 
                      annotation_position="top left")
        fig.add_vline(x=p5_investment, line_dash="dash", line_color="green",
                      annotation_text=f"5th: ${p5_investment:,.0f}", 
                      annotation_position="top right")
        fig.add_vline(x=p95_investment, line_dash="dash", line_color="green",
                      annotation_text=f"95th: ${p95_investment:,.0f}", 
                      annotation_position="top right")
    
    fig.update_layout(
        title=f"Additional Investment Required to Breakeven ({SIM_RUNS} Runs)",
        xaxis_title="Capital Investment Needed ($)",
        yaxis_title="Count / Frequency",
        height=450,
        margin={'t': 50, 'b': 20},
        showlegend=True
    )
    
    # Add a unique suffix to the key to avoid duplicates
    import hashlib
    key_suffix = hashlib.md5(scenario_name.encode()).hexdigest()[:8]
    tab.plotly_chart(fig, use_container_width=True, key=f"investment_dist_{key_suffix}")
    
    # Display summary metrics
    col1, col2, col3, col4 = tab.columns(4)
    
    if len(investment_breakeven) > 0:
        col1.metric("Median Investment (Breaks Even)", f"${np.median(investment_breakeven):,.0f}")
        col2.metric("Mean Investment (Breaks Even)", f"${np.mean(investment_breakeven):,.0f}")
        col3.metric("5th Percentile (Best Case)", f"${p5_investment:,.0f}")
        col4.metric("95th Percentile (Worst Case)", f"${p95_investment:,.0f}")
    else:
        col1.metric("Median Investment (Breaks Even)", "N/A")
        col2.metric("Mean Investment (Breaks Even)", "N/A")
        col3.metric("5th Percentile", "N/A")
        col4.metric("95th Percentile", "N/A")
    
    tab.markdown("---")

def display_single_run_trajectory(tab, all_scenario_results, scenario_name, run_index, sim_months):
    """
    Displays the monthly trajectory for a single selected MC run using Plotly. 
    Also includes the full results table at the end.
    """
    
    results = all_scenario_results[scenario_name]
    months = np.arange(1, sim_months + 1)
    run_data = {col: results[col][run_index, :] for col in results.keys()}
    
    tab.subheader(f"Single Run Deep Dive: Run {run_index+1}")
    
    df_plot = pd.DataFrame(run_data, index=months).reset_index().rename(columns={'index': 'Month'})
    df_plot_long = df_plot.melt(
        id_vars=['Month'], 
        value_vars=[col for col in results.keys()],
        var_name='Item', 
        value_name='Value'
    )
    
    AGGREGATE_LINES = ['Income', 'Payroll', 'Marketing', 'Overhead']
    
    df_master = df_plot_long[df_plot_long['Item'] == 'Master Profit']
    df_agg = df_plot_long[(df_plot_long['Item'].isin(AGGREGATE_LINES))]
    df_comp = df_plot_long[~df_plot_long['Item'].isin(AGGREGATE_LINES + ['Master Profit'])]
    
    fig = go.Figure()

    for item in df_comp['Item'].unique():
        df_item = df_comp[df_comp['Item'] == item]
        fig.add_trace(go.Scatter(
            x=df_item['Month'], y=df_item['Value'], mode='lines+markers', 
            name=item, line=dict(dash='dot', width=1), opacity=0.7,
            hovertemplate='Month: %{x}<br>Item: ' + item + '<br>Value: $%{y:,.0f}<extra></extra>'
        ))

    for item in df_agg['Item'].unique():
        df_item = df_agg[df_agg['Item'] == item]
        color = AGGREGATE_COLORS.get(item, 'gray')
        fig.add_trace(go.Scatter(
            x=df_item['Month'], y=df_item['Value'], mode='lines+markers', 
            name=item, line=dict(width=2, color=color), opacity=0.9,
            hovertemplate='Month: %{x}<br>Item: ' + item + '<br>Value: $%{y:,.0f}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=df_master['Month'], y=df_master['Value'], mode='lines+markers', 
        name='Master Profit', line=dict(color='black', width=4),
        hovertemplate='Month: %{x}<br>Profit: $%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Run {run_index+1} - Monthly Projection and Master Profit ({sim_months} Months)",
        xaxis_title="Month",
        yaxis_title="Value ($)",
        height=600,
        margin={'t': 50, 'b': 20}
    )
    
    tab.plotly_chart(fig, use_container_width=True, key=f"single_run_{scenario_name}_{run_index}")
    
    # --- 2. Summary Metrics ---
    tab.subheader("Run Summary Metrics")
    
    total_profit = run_data['Master Profit'].sum()
    
    col1, col2, col3, col4 = tab.columns(4)
    
    col1.metric(f"Total Profit ({sim_months} Mos)", f"${total_profit:,.0f}")
    col2.metric(f"Total Income ({sim_months} Mos)", f"${run_data['Income'].sum():,.0f}")
    col3.metric("Total Expenses", f"${(run_data['Marketing'] + run_data['Payroll'] + run_data['Overhead']).sum():,.0f}")
    col4.metric("Avg Monthly Profit", f"${np.mean(run_data['Master Profit']):,.0f}")

    # =======================================================
    # FEATURE: Full Results Table
    # =======================================================
    tab.header("Individual Run Performance Table")
    
    all_runs_data = all_scenario_results[scenario_name] 

    # Calculate Total Profit and Individual Metrics for all SIM_RUNS
    total_profit_all_runs = all_runs_data['Master Profit'].sum(axis=1)
    
    df_runs = pd.DataFrame({
        'Run #': np.arange(SIM_RUNS) + 1,
        'Total Master Profit': total_profit_all_runs,
        f'Total Income ({sim_months}M)': all_runs_data['Income'].sum(axis=1),
        f'Total Expenses ({sim_months}M)': (all_runs_data['Marketing'] + all_runs_data['Payroll'] + all_runs_data['Overhead']).sum(axis=1),
        'Avg Monthly Profit': np.mean(all_runs_data['Master Profit'], axis=1)
    }).sort_values(by='Total Master Profit', ascending=False).reset_index(drop=True)

    # Calculate Percentile Rank
    df_runs['Percentile Rank'] = df_runs['Total Master Profit'].rank(method='average', pct=True) * 100
    
    # Reorder columns and format
    df_runs = df_runs[['Run #', 'Percentile Rank', 'Total Master Profit', 
                       f'Total Income ({sim_months}M)', f'Total Expenses ({sim_months}M)', 'Avg Monthly Profit']]
    
    # Format the DataFrame for display
    styled_df = df_runs.style.format({
        'Total Master Profit': "${:,.0f}",
        f'Total Income ({sim_months}M)': "${:,.0f}",
        f'Total Expenses ({sim_months}M)': "${:,.0f}",
        'Avg Monthly Profit': "${:,.0f}",
        'Percentile Rank': "{:.1f}%"
    }).bar(subset=['Total Master Profit'], color='#5c8fbf', align='left') 

    tab.dataframe(styled_df, use_container_width=True, height=700)
    
# --- Main App ---

def main():
    st.set_page_config(layout="wide", page_title="P&L Monte Carlo Scenario Simulator")
    st.title("P&L Monte Carlo Scenario Simulator")
    
    # MODIFICATION: Use the DATA_FILE_NAME constant directly
    file_path = DATA_FILE_NAME
    df, reg_data, main_pl_columns, x_column = load_data_and_calculate_regressions(file_path)

    if df is None:
        st.stop()
        
    reg_model_options = list(MODEL_OPTIONS.keys())
    
    # --- 2. Sidebar Controls ---
    st.sidebar.header("Simulation Parameters")
    
    analysis_mode = st.sidebar.radio(
        "Select Modeling Approach",
        options=['Basic (Univariate)', 'Advanced (Composite)'],
        key='analysis_mode_select',
        help="Basic: Income/Payroll/Marketing/Overhead are projected directly (using only Agent Count). Advanced: Income/Payroll are projected as sums of component lines."
    )
    
    sim_months = st.sidebar.slider(
        "Number of Months to Simulate", 
        min_value=1, 
        max_value=36, 
        value=12, 
        step=1,
        key='sim_months_slider'
    )
    
    # Persistent State for Fits
    if 'selected_fits' not in st.session_state:
        all_cols_to_fit = ['Advances', 'Chargebacks', 'OtherIncome', 'Marketing', 
                           'PayrollSales', 'PayrollAdmin', 'PayrollOther', 'Overhead', 
                           'Income', 'Payroll']
        
        st.session_state['selected_fits'] = {col: 'Linear (1st Degree)' for col in all_cols_to_fit}
    
    fixed_scenarios = define_scenarios(CURRENT_AGENT_COUNT, sim_months)
    
    # --- Determine the columns to display/fit in TAB 1 based on mode ---
    if analysis_mode == 'Basic (Univariate)':
        fit_cols_to_show = ['Income', 'Marketing', 'Payroll', 'Overhead']
    else:
        fit_cols_to_show = ['Advances', 'Chargebacks', 'OtherIncome', 'Marketing', 
                            'PayrollSales', 'PayrollAdmin', 'PayrollOther', 'Overhead']

    # --- 4. Tab Structure ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Model Selection & Historical View",
        "2. Scenario 1 (Maintain)", 
        "3. Scenario 2 (Layoffs)", 
        "4. Scenario 3 (Hiring)",
        "5. Custom Scenario"
    ])

    # =================================================================
    # TAB 1: Independent Model Selection & Dedicated Charts
    # =================================================================
    with tab1:
        st.header(f"Historical Regression Analysis: {analysis_mode} Fit Selection")
        st.markdown("Select the best-fit model for **each P&L item**. Hover over a data point to see its original data.")
        
        pl_tabs = st.tabs(fit_cols_to_show)
        
        for i, col in enumerate(fit_cols_to_show):
            with pl_tabs[i]:
                st.subheader(f"Historical Data and Fit for {col}")
                
                # Determine the primary driver for the plot/fit
                if analysis_mode == 'Basic (Univariate)':
                    driver_col = 'Agent Count'
                elif col in ['Advances', 'Chargebacks', 'OtherIncome', 'Marketing']:
                    driver_col = 'AgentsPerforming'
                else:
                    driver_col = 'Agent Count'

                col_left, col_right = st.columns([1, 2])
                
                if col not in st.session_state['selected_fits']:
                    st.session_state['selected_fits'][col] = 'Linear (1st Degree)'

                with col_left:
                    # --- Find Best Model for the selected driver ---
                    models_for_driver = reg_data[col][driver_col]
                    best_model_name = max(models_for_driver, key=lambda k: models_for_driver[k]['r_squared'])
                    best_r2 = models_for_driver[best_model_name]['r_squared']

                    if st.session_state['selected_fits'][col] == best_model_name:
                          st.success(f"🥇 **Recommended Model:** {best_model_name} ($R^2$ {best_r2:.4f})")
                    else:
                          st.warning(f"🥇 Best Fit: {best_model_name} ($R^2$ {best_r2:.4f})")
                    
                    # --- Dropdown for Model Selection ---
                    fit_type = st.selectbox(
                        f"Select Regression Model for {col} vs {driver_col}:",
                        options=reg_model_options,
                        index=reg_model_options.index(st.session_state['selected_fits'][col]),
                        key=f'fit_select_{col}'
                    )
                    st.session_state['selected_fits'][col] = fit_type
                    
                    # --- Display Selected Model Stats ---
                    reg_details = reg_data[col][driver_col][fit_type]
                    
                    # Extract and display the regression equation
                    if 'Degree' in reg_details['model_name']:
                        # For polynomial models
                        degree = int(reg_details['model_name'].split('(')[-1].split('st')[0].split('nd')[0].split('rd')[0].strip())
                        
                        temp_df = df[[driver_col, col]].copy().dropna()
                        X_temp = temp_df[[driver_col]]
                        Y_temp = temp_df[col].values.reshape(-1, 1)
                        
                        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                        X_poly = poly_features.fit_transform(X_temp)
                        temp_model = LinearRegression()
                        temp_model.fit(X_poly, Y_temp)
                        
                        # Build equation string
                        intercept = temp_model.intercept_[0]
                        coeffs = temp_model.coef_[0]
                        
                        equation_parts = [f"{intercept:,.2f}"]
                        for i, coeff in enumerate(coeffs, start=1):
                            sign = "+" if coeff >= 0 else ""
                            if i == 1:
                                equation_parts.append(f"{sign}{coeff:.4f}·x")
                            else:
                                equation_parts.append(f"{sign}{coeff:.6f}·x^{i}")
                        
                        equation = f"y = {' '.join(equation_parts)}"
                        st.code(equation, language=None)
                    
                    elif 'Log' in reg_details['model_name'] or 'Ln' in reg_details['model_name']:
                        # For logarithmic models
                        temp_df = df[[driver_col, col]].copy().dropna()
                        X_temp = temp_df[[driver_col]]
                        Y_temp = temp_df[col].values.reshape(-1, 1)
                        
                        X_safe = np.maximum(X_temp.values, 1e-10)
                        if 'Log10' in reg_details['model_name']:
                            X_transformed = np.log10(X_safe)
                            log_label = "log₁₀(x)"
                        else:
                            X_transformed = np.log(X_safe)
                            log_label = "ln(x)"
                        
                        temp_model = LinearRegression()
                        temp_model.fit(X_transformed, Y_temp)
                        
                        intercept = temp_model.intercept_[0]
                        coeff = temp_model.coef_[0][0]
                        
                        sign = "+" if coeff >= 0 else ""
                        equation = f"y = {intercept:,.2f} {sign}{coeff:.4f}·{log_label}"
                        st.code(equation, language=None)
                        st.markdown(f"**Model Type:** {reg_details['model_name']}")
                    
                    else:
                        # For linear models
                        temp_df = df[[driver_col, col]].copy().dropna()
                        X_temp = temp_df[[driver_col]]
                        Y_temp = temp_df[col].values.reshape(-1, 1)
                        
                        model_class = MODEL_OPTIONS[fit_type]
                        temp_model = model_class()
                        temp_model.fit(X_temp, Y_temp)
                        
                        intercept = temp_model.intercept_[0]
                        coeff = temp_model.coef_[0][0]
                        
                        sign = "+" if coeff >= 0 else ""
                        equation = f"y = {intercept:,.2f} {sign}{coeff:.4f}·x"
                        st.code(equation, language=None)
                        st.markdown(f"**Model Type:** {reg_details['model_name']}")
                    
                    st.metric("R-Squared ($R^2$)", f"{reg_details['r_squared']:.4f}")
                    st.metric("Root Mean Sq. Error (RMSE)", f"${reg_details['rmse']:,.0f}")
                    st.metric("Standard Deviation ($\sigma$)", f"${reg_details['sigma']:,.0f}")

                with col_right:
                    fig = plot_historical_fit(df, col, reg_data, fit_type, driver_col)
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_fit_{col}")
    
    # --- 5. Run Monte Carlo (Live Update) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Status")
    scenario_results = run_monte_carlo_scenarios(reg_data, st.session_state['selected_fits'], fixed_scenarios, sim_months, analysis_mode)
    st.sidebar.success(f"MC Complete! ({sim_months} Months)")
    
    # --- 6. Monte Carlo Run Selector (Sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Individual Run Selection")
    run_options = [f"Run {i+1}" for i in range(SIM_RUNS)]
    selected_run_str = st.sidebar.selectbox(
        "Select Run to View in Scenario Tabs (2-5)", 
        run_options,
        key='run_selector'
    )
    selected_run_index = int(selected_run_str.split(' ')[1]) - 1

    # =================================================================
    # TABS 2, 3, 4: Fixed Scenarios Analysis
    # =================================================================
    
    def render_fixed_scenario_tab(tab, scenario_name, scenario_key):
        with tab:
            st.header(f"Scenario Analysis: {scenario_name}")
            
            agent_counts = fixed_scenarios[scenario_key]
            plot_agent_trajectory(tab, agent_counts, sim_months, scenario_name)

            plot_aggregate_pdf(tab, scenario_results, scenario_key, sim_months)
            
            # Add breakeven distribution chart
            plot_breakeven_distribution(tab, scenario_results, scenario_key, sim_months)

            display_single_run_trajectory(
                tab, 
                scenario_results, 
                scenario_key, 
                selected_run_index,
                sim_months
            )
            
    render_fixed_scenario_tab(tab2, "Scenario 1: Maintain Headcount (105)", "Scenario 1: Maintain Headcount (105)")
    render_fixed_scenario_tab(tab3, "Scenario 2: Lay off 3/mo", "Scenario 2: Lay off 3/mo")
    render_fixed_scenario_tab(tab4, "Scenario 3: Add 3/mo", "Scenario 3: Add 3/mo")


    # =================================================================
    # TAB 5: Custom Scenario
    # =================================================================
    with tab5:
        st.header("Custom Scenario Builder")
        st.markdown("Define a custom agent trajectory to model its impact on Master Profit.")

        col_a, col_b = st.columns(2)
        
        with col_a:
            custom_max_agents = st.number_input(
                "Max Agent Count (Ceiling)", 
                min_value=CURRENT_AGENT_COUNT, 
                value=150, 
                step=5,
                key='custom_max_agents'
            )
        with col_b:
            custom_rate = st.number_input(
                "Monthly Change in Agents (-5 to +5)", 
                min_value=-5, 
                max_value=15,
                value=2, 
                step=1,
                key='custom_rate'
            )
            
        # 1. Define custom agent count array
        custom_agents_ac = np.zeros(sim_months)
        current_agent = CURRENT_AGENT_COUNT
        for i in range(sim_months):
            current_agent += custom_rate
            custom_agents_ac[i] = min(max(current_agent, 10), custom_max_agents)
        
        final_custom_agents_ac = custom_agents_ac[-1]

        # Use the fixed ratio assumption for AgentsPerforming in the custom scenario as well
        last_performing_ratio = CURRENT_AGENT_COUNT / (CURRENT_AGENT_COUNT if CURRENT_AGENT_COUNT > 0 else 1)
        custom_agents_ap = np.minimum(custom_agents_ac * last_performing_ratio, custom_agents_ac).astype(int)

        custom_scenario_name = f"Custom: {custom_agents_ac[0]} -> {final_custom_agents_ac:.0f} Agents"
        custom_scenario_agent_counts = {
            custom_scenario_name: {'Agent Count': custom_agents_ac, 'AgentsPerforming': custom_agents_ap}
        }
        
        # 2. Run MC for custom scenario
        custom_results = run_monte_carlo_scenarios(reg_data, st.session_state['selected_fits'], custom_scenario_agent_counts, sim_months, analysis_mode)
        
        # 3. Display Custom Scenario Analysis
        
        custom_drivers = custom_scenario_agent_counts[custom_scenario_name]
        
        plot_agent_trajectory(tab5, custom_drivers, sim_months, custom_scenario_name)
        plot_aggregate_pdf(tab5, custom_results, custom_scenario_name, sim_months)
        
        # Add breakeven distribution chart
        plot_breakeven_distribution(tab5, custom_results, custom_scenario_name, sim_months)
        
        display_single_run_trajectory(
            tab5, 
            custom_results, 
            custom_scenario_name, 
            selected_run_index,
            sim_months
        )


if __name__ == '__main__':
    main()
