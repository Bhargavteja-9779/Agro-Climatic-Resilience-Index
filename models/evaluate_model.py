import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def determine_risk(yield_pred, yields_historical):
    """
    Risk logic: 
    High risk = predicted yield < 25th percentile of historical
    Medium risk = 25th - 75th percentile
    Low risk = > 75th percentile
    """
    p25 = np.percentile(yields_historical, 25)
    p75 = np.percentile(yields_historical, 75)
    
    risks = []
    for y in yield_pred:
        if y < p25:
            risks.append("High")
        elif y > p75:
            risks.append("Low")
        else:
            risks.append("Medium")
    return risks

def main():
    print("Evaluating models...")
    data_path = 'data/datasets/test_data.csv'
    if not os.path.exists(data_path):
        print("Test dataset not found. Run train_model.py first.")
        return
        
    df_test = pd.read_csv(data_path)
    X_test = df_test.drop(columns=['Yield'])
    y_test = df_test['Yield'].values
    
    # We load the full historical dataset to get yield percentiles for Risk mapping
    df_all = pd.read_csv('data/datasets/south_india_crop_data.csv')
    historical_yields = df_all['Yield'].values

    preprocessor = joblib.load('models/preprocessor.pkl')
    X_test_processed = preprocessor.transform(X_test)
    
    models = ['LinearRegression', 'RandomForest', 'GradientBoosting', 'XGBoost']
    
    results = []
    
    for name in models:
        model_path = f'models/{name}_model.pkl'
        if not os.path.exists(model_path):
            continue
            
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_processed)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Risk score calculation for demonstration
        risk_scores = determine_risk(y_pred, historical_yields)
        high_risk_count = risk_scores.count('High')
        
        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4),
            'High_Risk_Predictions': high_risk_count
        })
        
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    results_df = pd.DataFrame(results)
    
    os.makedirs('paper', exist_ok=True)
    results_df.to_csv('paper/model_comparison.csv', index=False)
    print("\nModel comparison saved to paper/model_comparison.csv")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
