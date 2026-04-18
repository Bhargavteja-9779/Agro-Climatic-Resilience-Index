import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('rocket')

def main():
    print("Running ACRI (Agro-Climatic Resilience Index) Sensitivity Experiment...")
    
    # Load Models and Data
    model_path = 'models/GradientBoosting_model.pkl'
    preprocessor_path = 'models/preprocessor.pkl'
    data_path = 'data/datasets/south_india_crop_data.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("Models not found. Train models first.")
        return
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    df_all = pd.read_csv(data_path)
    
    # Target Scenario: Rice in Coimbatore, Tamil Nadu
    target_crop = 'rice'
    target_state = 'Tamil Nadu'
    
    # Get historical 75th percentile (Q3) for Rice in Tamil Nadu
    hist_yields = df_all[(df_all['Crop'] == target_crop) & (df_all['State'] == target_state)]['Yield']
        
    q3_yield = np.percentile(hist_yields, 75)
    
    # Baseline biological requirements
    # Averages for Rice from the dataset
    baseline_req = df_all[df_all['Crop'] == target_crop].mean(numeric_only=True)
    optimal_rainfall = baseline_req['Rainfall_Required'] if 'Rainfall_Required' in baseline_req else 1000.0
    
    # Baseline Input for Model
    base_input = {
        'Crop': target_crop,
        'Season': 'Kharif',
        'State': target_state,
        'Area': 1000.0,
        'Fertilizer': 5000.0,
        'Pesticide': 200.0,
        'N': baseline_req['N'],
        'P': baseline_req['P'],
        'K': baseline_req['K'],
        'temperature': baseline_req['temperature'],
        'humidity': baseline_req['humidity'],
        'ph': baseline_req['ph'],
        'Rainfall_Required': optimal_rainfall
    }
    
    # Simulate Drought Severity (0% to 60% Deficit)
    deficits = np.linspace(0, 0.60, 20)  # 20 steps
    
    results = []
    
    for deficit in deficits:
        actual_rainfall = optimal_rainfall * (1 - deficit)
        delta_r = optimal_rainfall - actual_rainfall
        
        # Create single row DataFrame for prediction
        input_dict = base_input.copy()
        input_dict['Annual_Rainfall'] = actual_rainfall
        
        req_df = pd.DataFrame([input_dict])
        
        # Predict Yield
        X_processed = preprocessor.transform(req_df)
        predicted_yield = model.predict(X_processed)[0]
        
        # Calculate ACRI
        # ACRI = (Y_pred / Q3) * (1 - (|Delta R| / R_optimal))
        yield_ratio = predicted_yield / q3_yield
        climate_penalty = 1 - (abs(delta_r) / optimal_rainfall)
        acri_score = yield_ratio * climate_penalty
        
        results.append({
            'Drought Severity (%)': deficit * 100,
            'Predicted Yield (Tons/Ha)': predicted_yield,
            'ACRI Score': acri_score,
            'Rainfall (mm)': actual_rainfall
        })
        
    results_df = pd.DataFrame(results)
    
    # ---------------------------
    # Plotting the Experiment
    # ---------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary Axis (Predicted Yield)
    color1 = '#3498db'
    ax1.set_xlabel('Simulated Drought Severity (% Rainfall Deficit)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Yield (Tons/Ha)', color=color1, fontsize=12, fontweight='bold')
    line1, = ax1.plot(results_df['Drought Severity (%)'], results_df['Predicted Yield (Tons/Ha)'], 
             color=color1, linewidth=3, marker='o', label='Predicted Yield')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Secondary Axis (ACRI Score)
    ax2 = ax1.twinx()  
    color2 = '#e74c3c'
    ax2.set_ylabel('ACRI Resilience Score', color=color2, fontsize=12, fontweight='bold')
    line2, = ax2.plot(results_df['Drought Severity (%)'], results_df['ACRI Score'], 
             color=color2, linewidth=3, linestyle='--', marker='s', label='ACRI Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add Critical Threshold Line for ACRI
    ax2.axhline(y=0.4, color='#7f8c8d', linestyle=':', linewidth=2, label='Critical Failure Threshold (0.4)')
    ax2.fill_between(results_df['Drought Severity (%)'], 0, 0.4, alpha=0.1, color='red')
    
    # Title and Legend
    plt.title(f'ACRI Sensitivity Analysis: {target_crop.title()} in {target_state}', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends from both axes
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/acri_sensitivity_plot.png', dpi=300, bbox_inches='tight')
    print("Experiment successful. Saved ACRI Sensitivity Plot to visualizations/acri_sensitivity_plot.png")

if __name__ == "__main__":
    main()
