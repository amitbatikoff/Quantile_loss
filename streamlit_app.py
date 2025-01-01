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
def load_model(_trigger=False):  # Add trigger parameter
    input_size = 273  # Match the saved model architecture
    model = StockPredictor(input_size=input_size)
    model.load_state_dict(torch.load('unified_stock_model.pt'))
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
    
    # Plot historical data
    day_data = day_data[day_data['timestamp'].dt.date == selected_date]
    fig.add_trace(go.Scatter(
        x=day_data['timestamp'],
        y=day_data['close'],
        name='Daily Data',
        line=dict(color='lightgray', dash='dot')
    ))
    
    # Plot input data
    input_times = day_data['timestamp'].iloc[:len(input_data)]
    fig.add_trace(go.Scatter(
        x=input_times,
        y=input_data,
        name='Input Data',
        line=dict(color='blue')
    ))
    
    # Plot actual price
    target_time = day_data['timestamp'].iloc[int(len(day_data) * MODEL_PARAMS['target_ratio'])]
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
                            performance = float(predictions[1]) - last_price
                            performances.append((symbol, date, float(performance), float(last_price)))
                        except Exception as e:
                            continue
    
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
    
    try:
        stock_data = get_stock_data(SYMBOLS)
        _, _, test_data = split_data(stock_data)
        
        performances = calculate_stock_performance(model, test_data)
        top_10_predictions = performances[:10]
        
        # Updated formatting to handle the values more safely
        stock_options = []
        for symbol, date, perf, last_price in top_10_predictions:
            label = f"{symbol} on {date.strftime('%Y-%m-%d')} (P20 diff: ${perf:.2f} from ${last_price:.2f})"
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
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return

if __name__ == '__main__':
    main()
