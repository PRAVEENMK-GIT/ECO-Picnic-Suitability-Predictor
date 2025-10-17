from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and feature columns
model = joblib.load('best_picnic_model.pkl')
weather_df = pd.read_csv('weather_cleaned.csv')
feature_cols = [col for col in weather_df.columns if col not in ['DATE', 'MONTH', 'picnic']]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Fill missing features with mean values
    input_data = {col: data.get(col, float(weather_df[col].mean())) for col in feature_cols}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    return jsonify({'picnic_suitability': bool(prediction)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
