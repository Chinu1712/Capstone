from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("stock_price_lstm_model.h5")

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from JSON
        data = request.get_json(force=True)
        input_data = np.array(data['input']).reshape(1, 60, 1)  # Reshape for LSTM input
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Send response
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Main function to run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
