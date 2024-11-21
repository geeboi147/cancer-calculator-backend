import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all domains (you can restrict to specific origins later)
CORS(app)

# Load model and scaler files using relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Breast cancer models.pkl')

# Load the model and scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

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

        # Debugging: Print incoming data
        print(f"Received data: {data}")

        # Initialize the features list
        features = [
            float(data['age']),  # Age
            float(data['tumorSize']),  # Tumor Size
            float(data['invasiveNodes']),  # Invasive Nodes
            0 if data['breast'] == 'left' else 1,  # 'left' -> 0, 'right' -> 1
        ]

        # Map quadrant to numerical representation and extend features
        quadrant = quadrant_mapping.get(data['quadrant'], [0, 0, 0, 0])  # Default to [0, 0, 0, 0] if invalid
        print(f"Quadrant mapping for {data['quadrant']}: {quadrant}")  # Debugging output

        # Flatten the quadrant list and add to features
        features.extend(quadrant)

        # Add history and menopause as binary values
        features.append(0 if data['history'] == 'no' else 1)  # 'no' -> 0, 'yes' -> 1
        features.append(0 if data['menopause'] == 'no' else 1)  # 'no' -> 0, 'yes' -> 1

        # Log the final features
        print(f"Features passed to model: {features}")

        # Ensure features list is flat and no lists are nested
        assert all(isinstance(i, (int, float)) for i in features), "Features list contains non-numeric data"

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Get prediction and probability
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features).tolist()

        # Log prediction and probability
        print(f"Prediction: {prediction}, Probability: {probability}")

        # Return prediction and probabilities
        return jsonify({
            'prediction': int(prediction),  # Convert prediction to integer
            'probability': probability       # Return prediction probabilities
        })
    except Exception as e:
        # Handle errors and return them in the response
        print(f"Error: {str(e)}")  # Log error message
        return jsonify({'error': f"An error occurred: {str(e)}"}), 400

# Run the app with Gunicorn on Render (or locally with Flask's built-in server)
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
