import pandas as pd

# Load main weather dataset
weather_df = pd.read_csv('weather_prediction_dataset.csv')

# Load picnic labels
target_df = pd.read_csv('weather_prediction_picnic_labels.csv')

# Preview data
print('Weather Data:')
print(weather_df.head())
print('\nPicnic Labels:')
print(target_df.head())

# Check for missing values
print('\nMissing values in weather data:')
print(weather_df.isnull().sum())
print('\nMissing values in picnic labels:')
print(target_df.isnull().sum())

# Replace any remaining -9999 values with NaN, then fill or drop as appropriate
weather_df.replace(-9999, pd.NA, inplace=True)
weather_df = weather_df.fillna(weather_df.mean())

# Merge features and labels on common columns (e.g., date/location if available)
# For now, assume index alignment
if len(weather_df) == len(target_df):
    weather_df['picnic'] = target_df.iloc[:, -1]
else:
    print('Warning: Feature and label row counts do not match!')

# Save cleaned data for modeling
weather_df.to_csv('weather_cleaned.csv', index=False)
print('\nCleaned data saved to weather_cleaned.csv')
