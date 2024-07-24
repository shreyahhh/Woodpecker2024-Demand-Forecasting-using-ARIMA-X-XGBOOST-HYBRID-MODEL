import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import io
from statsmodels.graphics.tsaplots import plot_acf 


def process_chunk(chunk_df, arima_order):
    chunk_df['Date'] = pd.to_datetime(chunk_df['Date'])
    chunk_df.set_index('Date', inplace=True)
    chunk_df = chunk_df.groupby(chunk_df.index).agg('sum')
    chunk_df.sort_index(inplace=True)
    chunk_df = chunk_df.asfreq('D', fill_value=0)
    
    chunk_df['Lag1'] = chunk_df['Weekly_Sales'].shift(1)
    chunk_df['Lag2'] = chunk_df['Weekly_Sales'].shift(2)
    chunk_df = chunk_df.dropna()
    
    arima_model = ARIMA(chunk_df['Weekly_Sales'], order=arima_order)
    arima_result = arima_model.fit()
    
    chunk_df['ARIMA_Residuals'] = arima_result.resid
    
    return chunk_df, arima_result

def plot_residuals(residuals):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Residuals
    sns.lineplot(x=residuals.index, y=residuals, ax=ax[0])
    ax[0].set_title('ARIMA Residuals')
    
    max_lags = min(len(residuals) - 1, 40)  
    plot_acf(residuals, ax=ax[1], lags=max_lags)
    ax[1].set_title('Residuals ACF Plot')
    
    st.pyplot(fig)


def plot_predictions_vs_actual(chunk_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(chunk_df.index, chunk_df['Weekly_Sales'], label='Actual Sales')
    ax.plot(chunk_df.index, chunk_df['ARIMA_Residuals'] + chunk_df['Weekly_Sales'], label='Predicted Sales', linestyle='--')
    ax.set_title('Actual vs. Predicted Sales')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title('Demand Forecasting Model')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        chunk_df = pd.read_csv(uploaded_file)
        st.write("Data Sample:", chunk_df.head())
        
        # Separate sliders for ARIMA order (p, d, q)
        p = st.slider("Select ARIMA Order p", min_value=0, max_value=5, value=2)
        d = st.slider("Select ARIMA Order d", min_value=0, max_value=5, value=1)
        q = st.slider("Select ARIMA Order q", min_value=0, max_value=5, value=1)
        arima_order = (p, d, q)
        
        if st.button('Run Model'):
            st.write(f"Running ARIMA with order {arima_order}")
            chunk_df, arima_result = process_chunk(chunk_df, arima_order)
            
            st.write("Model Results:")
            mse = mean_squared_error(chunk_df['Weekly_Sales'].dropna(), arima_result.fittedvalues)
            st.write(f'MSE: {mse}')
            
            st.write("Residuals and Diagnostics:")
            plot_residuals(chunk_df['ARIMA_Residuals'].dropna())
            
            st.write("Predictions vs. Actual Sales:")
            plot_predictions_vs_actual(chunk_df)

            # Allow downloading results
            csv = chunk_df.to_csv(index=True)
            st.download_button("Download Results CSV", csv, "results.csv", "text/csv")


if __name__ == "__main__":
    main()

