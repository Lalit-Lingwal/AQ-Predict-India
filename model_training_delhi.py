import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. CONFIGURATION ---
DATA_FILE = os.path.join('data', 'city_day.csv') 

# --- CITIES TO TRAIN ---
# We will train a model for these cities.
# You can add or remove cities from this list.
CITIES_TO_TRAIN = [
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
N_FEATURES = 9 # We are using all 9 features
TARGET_VARIABLE = "PM2.5"
FINAL_FEATURE_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2',
    'NOx', 'NH3', 'CO', 'SO2', 'O3'
]

def load_and_process_city_data(df_full, city_name):
    """
    Filters the full dataframe for one city and processes it.
    """
    print(f"\nProcessing data for '{city_name}'...")
    
    df_city = df_full[df_full['City'] == city_name].copy()
    
    df_city.set_index('Date', inplace=True)
    df_city.sort_index(inplace=True)
    
    df_features = df_city[FINAL_FEATURE_COLUMNS].copy()
    
    # Fill small gaps
    df_features = df_features.ffill().bfill()
    # Drop any rows that are *still* NaN (which means the *entire* city has no data for that feature)
    df_features.dropna(inplace=True) 

    # Check if we have enough data to train (30 days lookback + 100 days to learn)
    if len(df_features) < (N_STEPS + 100): 
        print(f"Skipping '{city_name}': Not enough continuous data.")
        return None # Return None if not enough data
    
    # Reorder columns
    df_final = df_features[[TARGET_VARIABLE] + [col for col in FINAL_FEATURE_COLUMNS if col != TARGET_VARIABLE]]
    
    print(f"Data for '{city_name}' processed. Shape: {df_final.shape}")
    return df_final

def create_sequences(data, n_steps):
    """Creates sliding window sequences for LSTM."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        sequence_x, sequence_y = data[i:end_ix, :], data[end_ix, 0]
        X.append(sequence_x)
        y.append(sequence_y)
    return np.array(X), np.array(y)

def train_model_for_city(df_features, city_name):
    """
    Trains and saves a model for a single city.
    """
    print(f"\n--- Training Model for {city_name} ---")
    
    scaler_file = f'scaler_{city_name}.pkl'
    model_file = f'model_{city_name}.h5'
    
    # --- 1. Scale Data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_features)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file}")

    # --- 2. Create Sequences ---
    X, y = create_sequences(scaled_data, N_STEPS)
    
    if len(X) == 0:
        print(f"--- ERROR: No sequences created for {city_name}. Skipping. ---")
        return

    split_point = int(0.8 * len(X))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
        
    print(f"Created {len(X_train)} training samples and {len(X_test)} testing samples for {city_name}.")

    # --- 3. Build and Train Model ---
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(N_STEPS, N_FEATURES)),
        Dense(1) 
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary() 

    print(f"Training model for {city_name}...")
    model.fit(X_train, y_train, 
              epochs=30, 
              batch_size=32, 
              validation_data=(X_test, y_test), 
              verbose=1)
    
    print(f"Saving model to {model_file}...")
    model.save(model_file)
    print(f"--- ✅ SUCCESS! Model for {city_name} complete! ---")

def main():
    """
    Main script to loop through cities and train models.
    """
    print(f"Loading main data file from {DATA_FILE}...")
    try:
        df_full = pd.read_csv(DATA_FILE, parse_dates=['Date']) 
    except FileNotFoundError:
        print(f"--- ERROR: File not found: {DATA_FILE} ---")
        print("Please download the original 'city_day.csv' file and put it in the 'data' folder.")
        return

    for city in CITIES_TO_TRAIN:
        df_city_data = load_and_process_city_data(df_full, city)
        
        # --- THIS IS THE FIX ---
        # We must check if df_city_data is not None before passing it to the trainer
        if df_city_data is not None:
            train_model_for_city(df_city_data, city)
        
    print("\n\nAll cities trained successfully!")

if __name__ == '__main__':
    main()