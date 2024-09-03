from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError as e:
    raise RuntimeError("Model or scaler file not found. Please check the file paths.") from e
except pickle.PickleError as e:
    raise RuntimeError("Error loading model or scaler. Please check the file contents.") from e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Define the expected order of features
    feature_names = ['Specific_ailments', 'Age', 'BMI', 'Living_in', 'Follow_Diet',
                     'Physical_activity', 'Regular_sleeping_hours', 'Alcohol_consumption',
                     'Social_interaction', 'Illness_count_last_year', 'DX1', 'DX2', 'DX3',
                     'DX4', 'DX5', 'DX6', 'Smoker_NO', 'Smoker_YES']

    try:
        # Extract data from form in the correct order
        form_values = [request.form.get(name) for name in feature_names]
        int_features = [float(x) for x in form_values]
        final_features = np.array(int_features).reshape(1, -1)
        
        # Preprocessing (if required)
        final_features = scaler.transform(final_features)
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Healthy' if prediction[0] == 1 else 'Unhealthy'
        
        # Debugging output
        print(f"Prediction: {output}")
        
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text='Error occurred during prediction.')

if __name__ == "__main__":
    app.run(debug=True)
