# Import libraries
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
model = load_model('stock_price_lstm_model.h5')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        print(f"Received data: {data}")  # Debugging log
        prediction = model.predict(np.array(data).reshape(1, 60, 1))
        print(f"Prediction: {prediction}")  # Debugging log
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



