import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import torch
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, MODEL_PARAMS
import numpy as np
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
import plotly.express as px

@st.cache_resource
def load_model(_trigger=False):
    """Load model with architecture validation"""
    try:
        # First try to get input size from saved model
        state_dict = torch.load('unified_stock_model.pt', map_location=torch.device('cpu'))
        input_size = state_dict['model.1.weight'].shape[1]  # Get input size from first layer
    except Exception as e:
        st.error(f"Error detecting input size: {str(e)}")
        input_size = MODEL_PARAMS['architecture']['default_input_size']
        st.warning(f"Using fallback input size: {input_size}")

    try:
        model = StockPredictor(input_size=input_size)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Initializing new model with current architecture...")
        model = StockPredictor(input_size=input_size)
        model.eval()
        return model

@st.cache_data
def load_stock_data():
    stock_data = get_stock_data(SYMBOLS)
    train, val, test = split_data(stock_data)
    return train, val, test

def prepare_input_data(dataset, index):
    input_data, actual_price = dataset[index]
    actual_price += input_data[-1].item()
    return input_data.numpy(), actual_price.item(), dataset.data

def create_prediction_plot(day_data, input_data, actual_price, predictions, selected_date):
    fig = go.Figure()
    
    # Convert dictionary to DataFrame if necessary
    if isinstance(day_data, dict):
        # Get the data for the selected stock (first key in dictionary)
        stock_symbol = list(day_data.keys())[0]
        day_data = day_data[stock_symbol]
        
    # Filter data for selected date
    day_data = day_data[day_data['date'] == selected_date].copy()
    day_data.reset_index(inplace=True)
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=day_data['index'],
        y=day_data['close'],
        name='Daily Data',
        line=dict(color='lightgray', dash='dot')
    ))
    
    # Plot input data - fixed alignment to start from beginning of day
    input_times = day_data['index'].iloc[:len(input_data)]
    input_values = [float(x) for x in input_data]
    
    fig.add_trace(go.Scatter(
        x=input_times,
        y=input_values,
        name='Input Data',
        line=dict(color='blue', width=3),
        mode='lines+markers',
        marker=dict(size=8, color='blue'),
        opacity=1
    ))
    
    # Plot actual price and predictions
    target_idx = int(len(day_data) * MODEL_PARAMS['target_ratio'])
    target_time = day_data['index'].iloc[target_idx]
    
    # Plot actual price
    fig.add_trace(go.Scatter(
        x=[target_time],
        y=[actual_price],
        name='Actual Price',
        mode='markers',
        marker=dict(color='green', size=10)
    ))
    
    # Plot predicted quantiles
    colors = px.colors.qualitative.Set3
    for i, quantile in enumerate(MODEL_PARAMS['quantiles']):
        fig.add_trace(go.Scatter(
            x=[target_time],
            y=[predictions[i]],
            name=f'P{int(quantile*100)}',
            mode='markers',
            marker=dict(color=colors[i % len(colors)], size=8)
        ))
    
    fig.update_layout(
        title=f'Stock Price Prediction for {selected_date}',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

def calculate_stock_performance(model, test_data):
    performances = []
    
    for symbol in SYMBOLS:
        if symbol in test_data:
            dataset = StockDataset({symbol: test_data[symbol]})
            if len(dataset) > 0:
                for idx, (date, sym) in enumerate(dataset.date_symbols):
                    if sym == symbol:
                        try:
                            input_values, actual_price, _ = prepare_input_data(dataset, idx)
                            
                            with torch.no_grad():
                                input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
                                predictions = model(input_tensor)
                                predictions = predictions.squeeze().numpy() + input_values[-1]
                            
                            # Convert numpy values to Python native types
                            last_price = float(input_values[-1])
                            min_prediction = float(np.min(predictions))  
                            performance = (min_prediction - last_price)/max(0.1, last_price)
                            performances.append((symbol, date, performance, last_price))
                        except Exception as e:
                            print(f"Error calculating performance for {symbol} on {date}: {str(e)}")
                            raise
    
    return sorted(performances, key=lambda x: x[2], reverse=True)

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
    
    stock_data = get_stock_data(SYMBOLS)
    _, _, test_data = split_data(stock_data)
    
    performances = calculate_stock_performance(model, test_data)
    top_10_predictions = performances[:10]
    
    # Updated formatting to handle the values more safely
    stock_options = []
    for symbol, date, perf, last_price in top_10_predictions:
        label = f"{symbol} on {date.strftime('%Y-%m-%d')} (P10 chnage: {perf/last_price*100:.2f}%)"
        stock_options.append((label, (symbol, date)))
    
    if not stock_options:
        st.error("No stocks available for prediction")
        return
        
    selected_option = st.sidebar.radio(
        "Select from top 10 performing predictions",
        options=stock_options,
        format_func=lambda x: x[0]
    )
    selected_stock, selected_date = selected_option[1]
    
    # Modified the rest of the main function to use the selected date directly
    if selected_stock in test_data:
        test_df = test_data[selected_stock]
        if test_df.empty:
            st.error(f"No data available for {selected_stock}")
            return
            
        # Create dataset
        dataset = StockDataset({selected_stock: test_df})
        
        # Use the selected date directly instead of showing date picker
        if (selected_date, selected_stock) in dataset.date_symbols:
            # Prepare input data
            index = dataset.date_symbols.index((selected_date, selected_stock))
            input_values, actual_price, day_data = prepare_input_data(dataset, index)
            
            # Get prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
                predictions = model(input_tensor)
                predictions = predictions.squeeze().numpy() + input_values[-1]
            
            # Create and display the plot
            fig = create_prediction_plot(
                day_data,
                input_values,
                actual_price,
                predictions,
                selected_date
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction details in an expandable section
            with st.expander("View Quantile Predictions"):
                cols = st.columns(3)
                for i, quantile in enumerate(MODEL_PARAMS['quantiles']):
                    col_idx = i % 3
                    cols[col_idx].metric(
                        f"P{int(quantile*100)} Prediction", 
                        f"${predictions[i]:.2f}",
                        f"{((predictions[i] - actual_price) / actual_price * 100):.1f}%"
                    )
            
            # Display main metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Actual Price", f"${actual_price:.2f}")
            col2.metric("Median (P50)", f"${predictions[4]:.2f}")  # 0.5 quantile
            col3.metric("Prediction Interval", 
                        f"${predictions[0]:.2f} - ${predictions[-1]:.2f}")  # P10-P90
        else:
            st.warning('No data available for the selected date.')
    else:
        st.error(f"Failed to load test data for {selected_stock}")
        return


if __name__ == '__main__':
    main()
