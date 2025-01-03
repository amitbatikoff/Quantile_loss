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
import os
import glob
import debugpy


@st.cache_resource
def load_model(_trigger=False):
    """Load model with architecture validation"""
    # First try to get input size from saved model
    state_dict = torch.load('unified_stock_model.pt', map_location=torch.device('cpu'))
    input_size = state_dict['model.1.weight'].shape[1]  # Get input size from first layer
    
    # Initialize model with detected input size
    model = StockPredictor(input_size=int(input_size))  # Ensure input_size is int
    model.load_state_dict(state_dict)
    model.eval()
    
    # Verify initialization
    if not hasattr(model, '_input_size'):
        raise AttributeError("Model initialization failed: _input_size not set")
    
    return model
        

@st.cache_data
def load_stock_data():
    stock_data = get_stock_data(SYMBOLS)
    _, _, test_data = split_data(stock_data)
    return test_data

def prepare_input_data(dataset, index):
    try:
        input_data, actual_price = dataset[index]
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data)
        if not isinstance(actual_price, torch.Tensor):
            actual_price = torch.tensor(actual_price)
        
        actual_price += input_data[-1].item()
        return input_data.numpy(), actual_price.item(), dataset.data
    except Exception as e:
        st.error(f"Error preparing input data: {str(e)}")
        return None, None, None

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

@st.cache_data
def calculate_stock_performance(_model, test_data):
    if not hasattr(_model, '_input_size'):
        st.error("Invalid model: missing input_size attribute")
        return []
        
    performances = []
    
    for symbol in SYMBOLS:
        if symbol in test_data:
            try:
                # Use 'train' mode for prediction dataset
                pred_dataset = StockDataset({symbol: test_data[symbol]}, mode='train')
                # Validate input dimensions
                sample_input, _ = pred_dataset[0]
                if sample_input.shape[0] != _model._input_size:  # Access protected attribute
                    continue
                
                # Use 'viz' mode for visualization dataset
                viz_dataset = StockDataset({symbol: test_data[symbol]}, mode='viz')
                
                if len(viz_dataset) > 0:
                    for idx, (date, sym) in enumerate(viz_dataset.date_symbols):
                        if sym == symbol:
                            try:
                                # Get visualization data for reference
                                input_data = prepare_input_data(viz_dataset, idx)
                                if input_data[0] is None:
                                    continue
                                if  input_data[0].shape[0] != _model._input_size:
                                    continue
                                
                                input_values, actual_price,_ = input_data

                                # Find corresponding index in prediction dataset
                                pred_idx = pred_dataset.find_date_index(date)
                                if pred_idx is None:
                                    continue
                                    
                                # Get prediction using 'train' mode data
                                pred_input, _ = pred_dataset[pred_idx]
                                
                                with torch.no_grad():
                                    input_tensor = pred_input.unsqueeze(0)
                                    predictions = _model(input_tensor)
                                    predictions = predictions.squeeze().numpy() + input_values[-1]
                                
                                last_price = float(input_values[-1])
                                min_prediction = float(np.min(predictions))
                                performance = (min_prediction - last_price)/max(0.1, last_price)
                                performances.append((symbol, date, performance, last_price))
                            except Exception as e:
                                st.warning(f"Error processing {symbol} on {date}: {str(e)}")
                                continue
            except Exception as e:
                st.error(f"Error processing symbol {symbol}: {str(e)}")
                continue
    
    return sorted(performances, key=lambda x: x[2], reverse=True)

@st.cache_data(ttl=10)  # Cache for 10 seconds to allow for updates
def load_latest_metrics():
    try:
        # Find the latest metrics file in lightning_logs
        log_dirs = glob.glob("lightning_logs/version_*")
        if not log_dirs:
            return None
        
        latest_version = max(int(d.split('_')[-1]) for d in log_dirs)
        metrics_file = f"lightning_logs/version_{latest_version}/metrics.csv"
        
        if not os.path.exists(metrics_file):
            return None
            
        df = pd.read_csv(metrics_file)
        return df
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None

def plot_training_metrics(metrics_df):
    if metrics_df is None or metrics_df.empty:
        return None
        
    fig = go.Figure()
    
    # Plot training loss with explicit visibility
    if 'train_loss' in metrics_df.columns:
        filtered_df = metrics_df[metrics_df['train_loss'].notna()]
        fig.add_trace(go.Scatter(
            x=filtered_df['step'],
            y=filtered_df['train_loss'],
            name='Training Loss',
            mode='lines',
            line=dict(color='blue', width=2),
            visible=True
        ))
    
    # Plot validation loss with explicit visibility
    if 'val_loss' in metrics_df.columns:
        filtered_df = metrics_df[metrics_df['val_loss'].notna()]
        fig.add_trace(go.Scatter(
            x=filtered_df['step'],
            y=filtered_df['val_loss'],
            name='Validation Loss',
            mode='lines',
            line=dict(color='red', width=2),
            visible=True
        ))
    
    fig.update_layout(
        title='Training Metrics',
        xaxis_title='Steps',
        yaxis_title='Loss',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def main():
    # Sidebar controls
    st.sidebar.header('Select Parameters')
    
    if st.sidebar.button('Reload Model'):
        load_model.clear()
        st.sidebar.success('Model reloaded!')
    
    if st.sidebar.button('Attach debugger'):
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        st.title('Stock Price Prediction Visualization')

    # Load model with current trigger value
    model_loaded = True
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model_loaded = False

    # Create tabs
    tab1, tab2 = st.tabs(["Predictions", "Training Metrics"])

    # If model failed to load, automatically switch to tab 2
    if not model_loaded:
        tab1.error("Predictions tab disabled - Model failed to load")
        with tab2:
            st.header("Training Metrics")
            metrics_df = load_latest_metrics()
            
            if metrics_df is not None:
                fig = plot_training_metrics(metrics_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("View Raw Metrics Data"):
                        st.dataframe(metrics_df)
            else:
                st.warning("No training metrics found. Start training to see metrics here.")
        return

    # Rest of the code only executes if model loaded successfully
    with tab1:
        test_data = load_stock_data()
        performances = calculate_stock_performance(model, test_data)
        top_15_predictions = performances[:15]
        
        # Updated formatting to handle the values more safely
        stock_options = []
        for symbol, date, perf, last_price in top_15_predictions:
            label = f"{symbol} on {date.strftime('%Y-%m-%d')} (P10 chnage: {perf*100:.2f}%)"
            stock_options.append((label, (symbol, date)))
        
        if not stock_options:
            st.error("No stocks available for prediction")
            return
            
        selected_option = st.sidebar.radio(
            "Select from top 15 performing predictions",
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
                
            # Create datasets for both modes
            pred_dataset = StockDataset({selected_stock: test_df}, mode='train')
            viz_dataset = StockDataset({selected_stock: test_df}, mode='viz')
            
            if (selected_date, selected_stock) in viz_dataset.date_symbols:
                # Get visualization data
                viz_idx = viz_dataset.date_symbols.index((selected_date, selected_stock))
                input_values, actual_price, day_data = prepare_input_data(viz_dataset, viz_idx)
                
                # Get prediction data
                pred_idx = pred_dataset.find_date_index(selected_date)
                if pred_idx is not None:
                    pred_input, _ = pred_dataset[pred_idx]
                    
                    # Get prediction using train mode data
                    with torch.no_grad():
                        input_tensor = pred_input.unsqueeze(0)
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
                    
                    # Display main metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Last input", f"${input_values[-1]:.2f}")
                    col2.metric("Median (P50)", f"${predictions[4]:.2f}")  # 0.5 quantile
                    col3.metric("Prediction Interval", 
                                f"${predictions[0]:.2f} - ${predictions[-1]:.2f}")  # P10-P90

                    # Display main metrics
                    col1, col2 = st.columns(2)
                    col1.metric("P10 change", f"{100*(np.min(predictions)-input_values[-1])/input_values[-1]:.2f}%")
                    col2.metric("P50 change", f"{100*(np.median(predictions)-input_values[-1])/input_values[-1]:.2f}%")

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

                else:
                    st.warning('No data available for the selected date.')
            else:
                st.error(f"Failed to load test data for {selected_stock}")
                return

    with tab2:
        st.header("Training Metrics")
        
        # Add refresh button for metrics
        if st.button('Refresh Training Metrics'):
            load_latest_metrics.clear()
            st.success('Metrics refreshed!')
            
        metrics_df = load_latest_metrics()
        
        if metrics_df is not None:
            fig = plot_training_metrics(metrics_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Raw Metrics Data"):
                    st.dataframe(metrics_df)
        else:
            st.warning("No training metrics found. Start training to see metrics here.")

if __name__ == '__main__':
    main()
