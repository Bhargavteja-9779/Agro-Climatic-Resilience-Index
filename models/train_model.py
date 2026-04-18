import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

def main():
    print("Loading dataset...")
    data_path = 'data/datasets/south_india_crop_data.csv'
    if not os.path.exists(data_path):
        print("Dataset not found. Please run data_setup.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    # 'Yield' is the target. 'Production' is highly correlated but an absolute metric.
    # We drop 'Production' to avoid data leakage since Yield = Production/Area.
    target_col = 'Yield'
    
    # Optional Risk mapping logic: we calculate it for the frontend based on quantiles 
    # but we'll train the models to predict raw Yield.
    
    # Features
    categorical_features = ['Crop', 'Season', 'State', 'District']
    # Select available categorical columns
    cat_cols = [c for c in categorical_features if c in df.columns]
    
    numeric_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 
                        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'Rainfall_Required']
    num_cols = [c for c in numeric_features if c in df.columns]

    X = df[cat_cols + num_cols]
    y = df[target_col]

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Set sparse_output=False for SHAP compatibility later
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Defining models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples.")
    
    # Train and save each model
    os.makedirs('models', exist_ok=True)
    
    # Save the preprocessor separately so we can apply it to holdout sets for SHAP seamlessly
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # We will save the X_test and y_test sets for use by evaluation and explainability scripts
    pd.concat([X_test, y_test], axis=1).to_csv('data/datasets/test_data.csv', index=False)
    
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Note: We fit the model on PREPROCESSED data to decouple the steps for SHAP
        X_train_processed = preprocessor.transform(X_train)
        model.fit(X_train_processed, y_train)
        
        joblib.dump(model, f'models/{name}_model.pkl')
        print(f"Saved {name} to models/{name}_model.pkl")

if __name__ == "__main__":
    main()
