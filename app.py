from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/water_quality_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['ph']),
                float(request.form['hardness']),
                float(request.form['solids']),
                float(request.form['chloramines']),
                float(request.form['sulfate']),
                float(request.form['conductivity']),
                float(request.form['organic_carbon']),
                float(request.form['trihalomethanes']),
                float(request.form['turbidity'])
            ]

            # Scale and predict
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]

            prediction = '✅ Safe for Drinking (Potable)' if pred == 1 else '❌ Not Safe for Drinking'
        except:
            prediction = "⚠️ Invalid input. Please enter valid numbers."
    return render_template('index.html', prediction=prediction)


@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    predictions = None
    if request.method == 'POST':
        file = request.files['csv_file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)

            try:
                X = df[['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']]

                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)

                df['Potability_Prediction'] = preds
                df['Potability_Prediction'] = df['Potability_Prediction'].map({1: 'Potable', 0: 'Not Potable'})

                output_path = 'data/predicted_results.csv'
                df.to_csv(output_path, index=False)

                return send_file(output_path, as_attachment=True)
            except Exception as e:
                return f"Error in prediction: {e}"

    return render_template('batch_predict.html')

if __name__ == '__main__':
    app.run(debug=True)
