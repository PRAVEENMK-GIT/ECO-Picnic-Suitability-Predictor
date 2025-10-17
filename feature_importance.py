import joblib
import pandas as pd

# Load model and data
model = joblib.load('best_picnic_model.pkl')
weather_df = pd.read_csv('weather_cleaned.csv')
feature_cols = [col for col in weather_df.columns if col not in ['DATE', 'MONTH', 'picnic']]

# Get feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print(importance_df.head(20))
else:
    print('Model does not support feature importances.')
