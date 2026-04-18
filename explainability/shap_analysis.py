import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

def main():
    print("Running SHAP Explainability Analysis...")
    model_path = 'models/RandomForest_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Run train_model.py first.")
        return
        
    model = joblib.load(model_path)
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    # Load test data
    df_test = pd.read_csv('data/datasets/test_data.csv')
    X_test = df_test.drop(columns=['Yield'])
    
    # Transform test data to feed to model and SHAP
    X_test_processed = preprocessor.transform(X_test)
    
    # Retrieve feature names out of the ColumnTransformer
    cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    num_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([num_names, cat_names])
    
    # Ensure processed array is a DataFrame for SHAP visualization
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Create TreeExplainer (RandomForest is a tree-based model)
    # Using a sample to reduce computation time
    X_sample = shap.sample(X_test_df, 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('visualizations/shap_summary.png', dpi=300)
    print("Saved SHAP summary to visualizations/shap_summary.png")
    plt.close()
    
    # 2. SHAP Bar Plot (Global feature importance)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('visualizations/shap_feature_importance.png', dpi=300)
    print("Saved SHAP bar plot to visualizations/shap_feature_importance.png")
    plt.close()

if __name__ == "__main__":
    main()
