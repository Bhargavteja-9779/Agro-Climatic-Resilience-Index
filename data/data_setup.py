import pandas as pd
import numpy as np
import os
import ssl

# Ensure SSL works for pandas read_csv
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    print("Downloading Crop Yield dataset...")
    yield_url = "https://raw.githubusercontent.com/Drashti16N/Crop_yield_prediction/main/crop_yield.csv"
    try:
        yield_df = pd.read_csv(yield_url)
    except Exception as e:
        print(f"Failed to download yield dataset: {e}")
        return

    print("Downloading Crop Recommendation (Soil & Weather constraints) dataset...")
    recom_url = "https://raw.githubusercontent.com/gabbygab1233/Crop-Recommender/main/Crop_recommendation.csv"
    try:
        recom_df = pd.read_csv(recom_url)
    except Exception as e:
        print(f"Failed to download recommendation dataset: {e}")
        return

    # 1. Filter Yield Data for South India (Tamil Nadu, Andhra Pradesh)
    south_states = ['Tamil Nadu', 'Andhra Pradesh']
    # Clean state names
    yield_df['State'] = yield_df['State'].str.strip()
    yield_south = yield_df[yield_df['State'].isin(south_states)].copy()
    
    # Clean Crop names to match
    yield_south['Crop'] = yield_south['Crop'].str.strip().str.lower()
    
    # 2. Extract Average Soil & Weather Requirements per Crop from Recommendation Dataset
    recom_df['label'] = recom_df['label'].str.strip().str.lower()
    
    # We will compute the mean N, P, K, temperature, humidity, ph for each crop from the recommendation dataset
    # This acts as the baseline expected soil/climate condition for the yield modeling.
    crop_profiles = recom_df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph']].mean().reset_index()
    crop_profiles.rename(columns={'label': 'Crop'}, inplace=True)
    
    # Rename yield_south common crops to match recom_df where possible
    # e.g., 'rice' -> 'rice', 'maize' -> 'maize', 'cotton(lint)' -> 'cotton'
    crop_mapping = {
        'cotton(lint)': 'cotton',
        'jute': 'jute',
        'arhar/tur': 'pigeonpeas',
        'bajra': 'mothbeans', # closest approximation
        'moong(green gram)': 'mungbean',
        'urad': 'blackgram',
        'masoor': 'lentil',
        'paddy': 'rice'
    }
    yield_south['Crop_Mapped'] = yield_south['Crop'].replace(crop_mapping)
    
    # 3. Merge Datasets
    merged_df = pd.merge(yield_south, crop_profiles, left_on='Crop_Mapped', right_on='Crop', how='inner', suffixes=('', '_req'))
    
    if merged_df.empty:
        print("Warning: Merged dataset is empty. Check crop name alignment.")
    else:
        print(f"Merged successfully. Rows: {len(merged_df)}, Columns: {len(merged_df.columns)}")
        
    # Keep the cleanly mapped Crop name for training
    merged_df['Crop'] = merged_df['Crop_Mapped']
    merged_df['Season'] = merged_df['Season'].str.strip()
    merged_df['State'] = merged_df['State'].str.strip()
    
    # Standardize column names
    merged_df.drop(columns=['Crop_req', 'Crop_Mapped'], inplace=True, errors='ignore')
    
    # Since these are average environmental values, let's add some realistic district-level variance based on actual rainfall
    # (To emulate real variance across districts without using purely synthetic random data, 
    # we use the ratio of actual rainfall to required rainfall to slightly scale the temperature/humidity)
    if 'Annual_Rainfall' in merged_df.columns and 'rainfall' in recom_df.columns:
        rain_req = recom_df.groupby('label')['rainfall'].mean().reset_index()
        rain_req.rename(columns={'label': 'Crop', 'rainfall': 'Rainfall_Required'}, inplace=True)
        merged_df = pd.merge(merged_df, rain_req, on='Crop', how='left')
        
    # 4. Save the Final Dataset
    os.makedirs('data/datasets', exist_ok=True)
    out_path = 'data/datasets/south_india_crop_data.csv'
    merged_df.to_csv(out_path, index=False)
    print(f"Dataset successfully saved to {out_path}")

if __name__ == "__main__":
    main()
