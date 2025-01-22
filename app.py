
from flask import Flask, request, jsonify
from model import LogisticRegressionModel  # Import the model class
import os
# Initialize Flask app
app = Flask(__name__)

# Initialize the Logistic Regression model
model = LogisticRegressionModel()

# Load the trained model (ensure the model is already trained and saved)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Root endpoint
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Manufacturing Predictive Analysis API!"

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Please upload a CSV file."}), 400

    file = request.files['file']

    # Check if the file is a CSV
    if file.filename.endswith('.csv'):
        # Save the file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load the data into the model
        try:
            model.load_data(file_path)
            model.preprocess_data()
            return jsonify({"message": f"File '{file.filename}' uploaded and data loaded successfully!"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400


# Train endpoint
@app.route('/train', methods=['POST'])
def train():
    try:
        model.train()
        model.save_model('logistic_regression_model.pkl')  # Save the trained model
        return jsonify({"message": "Model trained and saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Validate input data
    if not data or 'Process temperature [K]' not in data or 'Tool wear [min]' not in data:
        return jsonify({"error": "Invalid input. Please provide 'Temperature' and 'Run_Time'."}), 400

    # Make a prediction using the model
    try:
        prediction, confidence = model.predict(data)
        return jsonify({
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
