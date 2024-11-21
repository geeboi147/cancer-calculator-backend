import os
import pickle
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all domains (you can restrict to specific origins later)
CORS(app, origins=["https://cancer-calculator.vercel.app/"])

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Load model and scaler files using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Breast cancer models.pkl')

# Load the model and scaler with error handling
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define mappings for categorical values
quadrant_mapping = {
    "upper_inner": [1, 0, 0, 0],
    "upper_outer": [0, 1, 0, 0],
    "lower_inner": [0, 0, 1, 0],
    "lower_outer": [0, 0, 0, 1],
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Validate required fields
        required_keys = ['age', 'tumorSize', 'invasiveNodes', 'breast', 'quadrant', 'history', 'menopause']
        for key in required_keys:
            if key not in data:
                logging.error(f"Missing key: {key}")
                return jsonify({'error': f"Missing key: {key}"}), 400

        # Initialize the features list
        features = [
            float(data['age']),  # Age
            float(data['tumorSize']),  # Tumor Size
            float(data['invasiveNodes']),  # Invasive Nodes
            0 if data['breast'] == 'left' else 1,  # 'left' -> 0, 'right' -> 1
        ]

        # Map quadrant to numerical representation and extend features
        quadrant = quadrant_mapping.get(data['quadrant'], [0, 0, 0, 0])  # Default to [0, 0, 0, 0] if invalid
        features.extend(quadrant)

        # Add history and menopause as binary values
        features.append(0 if data['history'] == 'no' else 1)  # 'no' -> 0, 'yes' -> 1
        features.append(0 if data['menopause'] == 'no' else 1)  # 'no' -> 0, 'yes' -> 1

        # Log the final features
        logging.debug(f"Features passed to model: {features}")

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Get prediction and probability
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features).tolist()

        # Log prediction and probability
        logging.debug(f"Prediction: {prediction}, Probability: {probability}")

        # Return prediction and probabilities
        return jsonify({
            'prediction': int(prediction),  # Convert prediction to integer
            'probability': probability       # Return prediction probabilities
        })
    except Exception as e:
        # Handle errors and return them in the response
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 400

# Serve a favicon to prevent 404 errors
@app.route('/favicon.ico')
def favicon():
    return jsonify({'message': 'No favicon available'}), 204

# Run the app with Gunicorn on Render (or locally with Flask's built-in server)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
