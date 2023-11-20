# Import necessary modules
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
# Create a Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the pre-trained models
model_main_meter = joblib.load('model_main_meter_dt.pkl')
model_resource_centre = joblib.load('model_resource_centre_dt.pkl')

# Prediction route for both Main Meter and Resource Centre
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model')
    features = data.get('features')

    if model_name is None or features is None:
        return jsonify({'error': 'Invalid input format'}), 400

    feature_value = features.get('Number of Students')

    if feature_value is None:
        return jsonify({'error': 'Missing feature value'}), 400

    X_test = pd.DataFrame({'Number of Students': [feature_value]})

    if model_name.lower() == 'main meter':
        prediction = model_main_meter.predict(X_test)[0]
    elif model_name.lower() == 'resource centre':
        prediction = model_resource_centre.predict(X_test)[0]
    else:
        return jsonify({'error': 'Invalid model name'}), 400

    # Include 'Number of Students' in the response
    response = {
        'model': model_name,
        'number_of_students': feature_value,
        'prediction': prediction
    }

    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)
