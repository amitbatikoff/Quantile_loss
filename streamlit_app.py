import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from predictor import StockPredictor
from data_loader import get_stock_data
from config import SYMBOLS, MODEL_PARAMS
import numpy as np

@st.cache_resource
def load_model(_trigger=False):  # Add trigger parameter
    input_size = 273  # Match the saved model architecture
    model = StockPredictor(input_size=input_size, hidden_dim=MODEL_PARAMS["hidden_dim"])
    model.load_state_dict(torch.load('unified_stock_model.pt'))
    model.eval()
    return model

@st.cache_data
def load_stock_data():
    return get_stock_data(SYMBOLS)

def prepare_input_data(df, date):
    day_data = df[df['timestamp'].dt.date == date].sort_values('timestamp')
    total_len = len(day_data)
    input_len = int(total_len * MODEL_PARAMS['input_ratio'])
    target_idx = int(total_len * MODEL_PARAMS['target_ratio'])
    
    input_data = day_data.iloc[:input_len]['close'].values
    actual_price = day_data.iloc[target_idx]['close']
    
    return input_data, actual_price, day_data

def create_prediction_plot(day_data, input_data, actual_price, predicted_price, selected_date):
    fig = go.Figure()
    
    # Plot all daily data points
    fig.add_trace(go.Scatter(
        x=day_data['timestamp'],
        y=day_data['close'],
        name='Daily Data',
        line=dict(color='lightgray', dash='dot')
    ))
    
    # Plot input data used for prediction
    input_times = day_data['timestamp'].iloc[:len(input_data)]
    fig.add_trace(go.Scatter(
        x=input_times,
        y=input_data,
        name='Input Data',
        line=dict(color='blue')
    ))
    
    # Plot actual and predicted prices
    target_time = day_data['timestamp'].iloc[int(len(day_data) * MODEL_PARAMS['target_ratio'])]
    fig.add_trace(go.Scatter(
        x=[target_time],
        y=[actual_price],
        name='Actual Price',
        mode='markers',
        marker=dict(color='green', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[target_time],
        y=[predicted_price],
        name='Predicted Price',
        mode='markers',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f'Stock Price Prediction for {selected_date}',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

def main():
    st.title('Stock Price Prediction Visualization')
    
    # Sidebar controls
    st.sidebar.header('Select Parameters')
    
    # Add reload model button
    if st.sidebar.button('Reload Model'):
        # Clear the cache for load_model
        load_model.clear()
        st.sidebar.success('Model reloaded!')
    
    # Load model with current trigger value
    model = load_model()
    
    # Rest of the main function remains the same
    stock_data = load_stock_data()
    
    selected_stock = st.sidebar.selectbox('Select Stock', SYMBOLS)
    
    # Get data for selected stock
    df = stock_data[selected_stock]
    
    # Group by date to ensure we have complete days
    daily_groups = df.groupby(df['timestamp'].dt.date)
    available_dates = sorted(daily_groups.groups.keys())
    
    # Filter dates that have enough data points
    valid_dates = [date for date in available_dates 
                  if len(daily_groups.get_group(date)) > 10]  # Minimum data points threshold
    
    if not valid_dates:
        st.error("No valid dates with sufficient data points found.")
        return
    
    selected_date = st.sidebar.date_input(
        'Select Date',
        min_value=valid_dates[0],
        max_value=valid_dates[-1],
        value=valid_dates[0]
    )

    if selected_date in valid_dates:
        # Prepare input data
        input_values, actual_price, day_data = prepare_input_data(df, selected_date)
        
        # Get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
            predicted_price = model(input_tensor).item()
        
        # Create and display the plot
        fig = create_prediction_plot(
            day_data,
            input_values,
            actual_price,
            predicted_price,
            selected_date
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display prediction details
        col1, col2, col3 = st.columns(3)
        col1.metric("Actual Price", f"${actual_price:.2f}")
        col2.metric("Predicted Price", f"${predicted_price:.2f}")
        col3.metric("Difference", f"${(predicted_price - actual_price):.2f}")
    else:
        st.warning('No data available for the selected date.')

if __name__ == '__main__':
    main()
