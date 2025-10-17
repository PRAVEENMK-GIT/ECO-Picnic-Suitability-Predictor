import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load cleaned data
weather_df = pd.read_csv('weather_cleaned.csv')

# Use all columns except 'DATE', 'MONTH', and 'picnic' as features
feature_cols = [col for col in weather_df.columns if col not in ['DATE', 'MONTH', 'picnic']]
X = weather_df[feature_cols]
y = weather_df['picnic']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
