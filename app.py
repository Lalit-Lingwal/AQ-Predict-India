import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from pandas.tseries.offsets import DateOffset
import os

# --- 1. CONFIGURATION ---
DATA_FILE = os.path.join('data', 'city_day.csv') 

# --- CITIES TO SUPPORT ---
# This list MUST match the cities you trained in the other script
CITIES = [
    'Delhi',
    'Mumbai',
    'Ahmedabad',
    'Bangalore',
    'Chennai',
    'Hyderabad',
    'Kolkata'
]

# --- Model Parameters ---
N_STEPS = 30
N_FEATURES = 9 
TARGET_VARIABLE = "PM2.5"
FINAL_FEATURE_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2',
    'NOx', 'NH3', 'CO', 'SO2', 'O3'
]

# --- 2. DATA AND MODEL LOADING ---

@st.cache_resource
def load_all_models_and_scalers(city_list):
    """
    Loads all models and scalers into a dictionary on startup.
    """
    models = {}
    scalers = {}
    print("Loading all models...")
    for city in city_list:
        model_file = f'model_{city}.h5'
        scaler_file = f'scaler_{city}.pkl'
        try:
            models[city] = tf.keras.models.load_model(model_file)
            with open(scaler_file, 'rb') as f:
                scalers[city] = pickle.load(f)
            print(f"Successfully loaded model for {city}.")
        except FileNotFoundError:
            print(f"Warning: Model or scaler file for {city} not found. This city will not be available.")
    
    return models, scalers

@st.cache_data
def load_and_process_data(filepath):
    """
    Loads the original city_day.csv file ONCE.
    """
    print(f"APP: Loading and processing data from {filepath}...")
    try:
        df = pd.read_csv(filepath, parse_dates=['Date']) 
    except FileNotFoundError:
        return None
    
    df_features_all_cities = {}
    for city in CITIES:
        print(f"APP: Processing data for {city}...")
        df_city = df[df['City'] == city].copy()
        df_city.set_index('Date', inplace=True)
        df_city.sort_index(inplace=True)
        
        df_features = df_city[FINAL_FEATURE_COLUMNS].copy()
        df_features = df_features.ffill().bfill()
        df_features.dropna(inplace=True) # Drop any remaining NaNs
        
        df_features_all_cities[city] = df_features
    
    print("APP: All city data processed.")
    return df_features_all_cities

# --- Load all assets ---
models, scalers = load_all_models_and_scalers(CITIES)
all_data = load_and_process_data(DATA_FILE)

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-City AQI Forecast",
    page_icon="💨",
    layout="wide"
)

# --- 4. MAIN APP ---
st.title("🇮🇳 Multi-City Air Quality Analysis & Forecasting")

# --- City Selection ---
st.sidebar.header("Select City")
# Filter list to only show cities that successfully loaded
available_cities = [city for city in CITIES if city in models and not all_data[city].empty]
if not available_cities:
    st.error("Fatal Error: No models or data could be loaded. Please run the training script first.")
    st.stop()

selected_city = st.sidebar.selectbox(
    "Choose a city to analyze:",
    options=available_cities
)

# --- Filter data and models based on selection ---
model = models[selected_city]
scaler = scalers[selected_city]
df = all_data[selected_city]

# --- KPI Section ---
st.header(f"Key Performance Indicators (KPIs) for {selected_city}")
avg_pm25 = df['PM2.5'].mean()
max_pm25 = df['PM2.5'].max()
max_pm25_day = df['PM2.5'].idxmax().strftime('%B %d, %Y')
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(label="Overall Avg. PM2.5", value=f"{avg_pm25:.2f} µg/m³")
kpi2.metric(label="Peak PM2.5 Recorded", value=f"{max_pm25:.2f} µg/m³", help=f"Highest level on {max_pm25_day}")
kpi3.metric(label="Total Days Logged", value=f"{len(df)} days")
st.markdown("---")

# --- Interactive Time Series Chart ---
st.header(f"📈 Pollutant Trends Over Time in {selected_city}")
df_weekly = df[['PM2.5', 'PM10', 'CO']].resample('W').mean()
fig_trends = px.line(df_weekly, title=f"Weekly Average Pollutant Concentrations in {selected_city}")
st.plotly_chart(fig_trends, use_container_width=True)
st.markdown("---")

# --- Deeper Analysis Section ---
st.header("🔬 Deeper Analysis")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Pollutant Correlation Heatmap")
    corr = df[FINAL_FEATURE_COLUMNS].corr()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_corr)
with col2:
    st.subheader("Monthly Pollution Patterns")
    monthly_avg = df.groupby(df.index.month).mean()
    monthly_avg.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig_monthly = px.bar(monthly_avg, y='PM2.5', title=f"Average PM2.5 by Month in {selected_city}")
    st.plotly_chart(fig_monthly, use_container_width=True)
st.markdown("---")

# --- Prediction Section ---
with st.expander("🔮 Predict Next Day's PM2.5 Level", expanded=True):
    st.header(f"Daily PM2.5 Prediction for {selected_city}")
    pred_col1, pred_col2 = st.columns([1, 2])
    
    with pred_col1:
        st.write("Select a date. The model will use the 30 days *before* this date to make its prediction.")
        
        # Date input constraints
        min_date = df.index[N_STEPS + 1].date()
        max_date = df.index[-1].date()
        
        selected_date = st.date_input("Select a date for prediction", 
                                      value=max_date, 
                                      min_value=min_date, 
                                      max_value=max_date)
        
        if st.button("Predict PM2.5"):
            selected_datetime = pd.to_datetime(selected_date)
            
            start_date = selected_datetime - DateOffset(days=N_STEPS)
            end_date = selected_datetime - DateOffset(days=1)
            
            # Check if we have this data in our dataframe
            if start_date not in df.index or end_date not in df.index:
                st.error("Error: The 30-day period before this date is not available or contains large data gaps. Please choose a different date.")
            else:
                last_30_days_data = df.loc[start_date : end_date]

                if len(last_30_days_data) == N_STEPS:
                    input_data = last_30_days_data[FINAL_FEATURE_COLUMNS] 
                    scaled_input_data = scaler.transform(input_data)
                    
                    reshaped_input = np.reshape(scaled_input_data, (1, N_STEPS, N_FEATURES))
                    
                    prediction_scaled = model.predict(reshaped_input)
                    
                    dummy_array = np.zeros((1, N_FEATURES))
                    
                    target_col_index = FINAL_FEATURE_COLUMNS.index(TARGET_VARIABLE)
                    dummy_array[0, target_col_index] = prediction_scaled[0, 0]
                    
                    prediction_actual_val = scaler.inverse_transform(dummy_array)[0, target_col_index]
                    
                    if selected_datetime in df.index:
                        actual_value = df.loc[selected_datetime][TARGET_VARIABLE]
                    else:
                        actual_value = np.nan 

                    st.session_state['prediction_data'] = {
                        'last_30_days': last_30_days_data, 
                        'predicted_value': prediction_actual_val,
                        'actual_value': actual_value, 
                        'prediction_point': selected_datetime
                    }
                    
                    st.metric(label=f"Predicted PM2.5 for {selected_date}", value=f"{prediction_actual_val:.2f} µg/m³")
                    st.metric(label=f"Actual PM2.5 on {selected_date}", value=f"{actual_value:.2f} µg/m³", help="This is the real value from the dataset, for comparison.")
                else:
                    st.error(f"Not enough historical data. Found {len(last_30_days_data)} days, but need {N_STEPS}.")

    with pred_col2:
        if 'prediction_data' in st.session_state:
            p_data = st.session_state['prediction_data']
            st.subheader("Prediction Context")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(p_data['last_30_days'].index, p_data['last_30_days'][TARGET_VARIABLE], label='Input Data (Last 30 Days)', marker='o', linestyle='-')
            
            ax.scatter(p_data['prediction_point'], p_data['predicted_value'], 
                       color='red', s=100, zorder=5, label=f"Predicted: {p_data['predicted_value']:.2f}")
            
            if not np.isnan(p_data['actual_value']):
                ax.scatter(p_data['prediction_point'], p_data['actual_value'], 
                           color='green', s=100, zorder=5, label=f"Actual: {p_data['actual_value']:.2f}")
            
            plt.xticks(rotation=45)
            ax.legend()
            ax.set_title("PM2.5 Levels: Input vs. Prediction")
            ax.set_ylabel("PM2.5 Concentration (µg/m³)")
            st.pyplot(fig)
        else:
            st.info("Click the 'Predict PM2.5' button to see the chart and results.")