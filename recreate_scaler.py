import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Paths
data_path = r'c:\Users\HP\Desktop\real estate\real estate\data.csv'
mfg_path = r'C:\Users\HP\Desktop\mfg'
market_stats_path = os.path.join(mfg_path, 'market_stats.pkl')

def create_scaler():
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # 1. Cleaning (matching notebook)
    df['date'] = pd.to_datetime(df['date'])
    allowed_cols = [
        'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
        'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
        'yr_built', 'yr_renovated', 'street', 'city', 'statezip', 'country'
    ]
    # Check if columns exist before selecting
    existing_cols = [c for c in allowed_cols if c in df.columns]
    df = df[existing_cols].dropna()

    # Outlier removal
    q_low = df["price"].quantile(0.01)
    q_hi  = df["price"].quantile(0.99)
    df = df[(df["price"] > q_low) & (df["price"] < q_hi)]

    # 2. Feature Engineering
    # Load city map from market_stats if available, else recompute (better to use saved to stay consistent)
    if os.path.exists(market_stats_path):
        stats = joblib.load(market_stats_path)
        city_price_map = stats.get('city_map')
        if city_price_map is None:
             # Fallback
             city_price_map = df.groupby('city')['price'].median().sort_values().to_dict()
    else:
        city_price_map = df.groupby('city')['price'].median().sort_values().to_dict()

    df['city_enc'] = df['city'].map(city_price_map)
    df['house_age'] = 2026 - df['yr_built']
    df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    # 3. Prepare X for Scaling
    # Features expected by model (from inspector output)
    expected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'city_enc', 'house_age', 'was_renovated']
    
    # Ensure all features exist
    for col in expected_features:
        if col not in df.columns:
            print(f"Warning: Expected feature {col} not in dataframe")
            # Handle potential missing? city_enc might be NaN if city not in map?
            # Impute or drop?
    
    df = df.dropna(subset=expected_features)
    
    X = df[expected_features]
    
    print("Fitting scaler...")
    scaler = StandardScaler()
    scaler.fit(X)
    
    save_path = os.path.join(mfg_path, 'scaler.pkl')
    joblib.dump(scaler, save_path)
    print(f"Scaler saved to {save_path}")
    print("Scaler mean:", scaler.mean_)
    print("Scaler scale:", scaler.scale_)

if __name__ == "__main__":
    create_scaler()
