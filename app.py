import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the trained model, imputer, and encoder
try:
    with open('storm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('storm_imputer.pkl', 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)
    with open('storm_encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
except FileNotFoundError:
    print("Error: Model or preprocessor files not found. Please run storm_model_fixed.py first.")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Extract features, handle missing pressure by setting it to NaN
        lat = float(data['lat'])
        long = float(data['long'])
        wind = float(data['wind'])
        pressure = data.get('pressure')
        pressure = float(pressure) if pressure else np.nan

        # Create a numpy array for the model
        features = np.array([[lat, long, wind, pressure]])

        # Impute missing values using the loaded imputer
        features_imputed = imputer.transform(features)

        # Make a prediction
        prediction_encoded = model.predict(features_imputed)
        
        # Decode the prediction to the original label
        prediction = encoder.inverse_transform(prediction_encoded)[0]

        # Return the result as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
