import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load cleaned data
weather_df = pd.read_csv('weather_cleaned.csv')

# Use all columns except 'DATE', 'MONTH', and 'picnic' as features
feature_cols = [col for col in weather_df.columns if col not in ['DATE', 'MONTH', 'picnic']]
X = weather_df[feature_cols]
y = weather_df['picnic']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print('Best parameters:', grid_search.best_params_)

# Evaluate best model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)
print('Optimized Accuracy:', accuracy_score(y_test, y_pred))
print('\nOptimized Classification Report:')
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_clf, 'best_picnic_model.pkl')
print('Best model saved as best_picnic_model.pkl')
