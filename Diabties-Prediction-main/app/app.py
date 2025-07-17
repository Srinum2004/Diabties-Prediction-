from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load(r'F:\Complete 3 Project\Diabties-Prediction-main\Diabties-Prediction-main\Notebook\diabetes_prediction_model.pkl')#always change this path according to your actuall path
scaler = joblib.load(r'F:\Complete 3 Project\Diabties-Prediction-main\Diabties-Prediction-main\Notebook\scaler.pkl')#always change this path according to your actuall path
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get gender and pregnancies
        gender = request.form['gender']
        pregnancies = float(request.form['pregnancies']) if gender == 'female' else 0.0
        
        # Prepare data
        data = {
            'Pregnancies': pregnancies,
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['bloodpressure']),
            'SkinThickness': float(request.form['skinthickness']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['pedigree']),
            'Age': float(request.form['age'])
        }
        
        # Predict
        input_data = pd.DataFrame([data])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]
        
        result = {
            'gender': gender,
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'probability': round(probability * 100, 2),
            'used_pregnancies': pregnancies
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)