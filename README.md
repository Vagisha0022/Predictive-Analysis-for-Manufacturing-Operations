# Predictive-Analysis-for-Manufacturing-Operations
Project Overview
This project implements a RESTful API using Flask to predict machine downtime based on manufacturing data. It uses a logistic regression model trained on data containing information like process temperature and tool wear to predict whether a machine will experience downtime.

The API provides endpoints for:

Uploading a dataset.
Training the model.
Making predictions based on input features.

Features
Upload Dataset: Upload a CSV file containing manufacturing data for training.
Train Model: Train the logistic regression model using the uploaded dataset.
Predict Downtime: Use the trained model to predict whether downtime will occur based on input features.

Tech Stack
Programming Language: Python
Framework: Flask
Machine Learning Library: scikit-learn
API Testing Tool: Postman

Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/manufacturing-predictive-analysis.git
cd manufacturing-predictive-analysis
2. Install Dependencies
Install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
3. Run the Flask App
Start the Flask application:

bash
Copy
Edit
python app/app.py
The app will run on http://127.0.0.1:5000.

Endpoints
1. Root Endpoint
URL: /
Method: GET
Description: Confirms that the API is running.
Example Response:
json
Copy
Edit
{
  "message": "Welcome to the Manufacturing Predictive Analysis API!"
}
2. Upload Dataset
URL: /upload
Method: POST
Description: Upload a CSV file containing the dataset.
Input:
A CSV file with the following columns:
Process temperature [K]
Tool wear [min]
Target (0 or 1 indicating no downtime or downtime).
Example Request in Postman:
Select POST.
Set the URL to http://127.0.0.1:5000/upload.
Go to the Body tab and select form-data.
Add a key named file, set it to File, and upload your CSV file.
Example Response:
json
Copy
Edit
{
  "message": "File 'predictive_maintenance.csv' uploaded and data loaded successfully!"
}
3. Train Model
URL: /train
Method: POST
Description: Trains the logistic regression model on the uploaded dataset.
Example Request in Postman:
Select POST.
Set the URL to http://127.0.0.1:5000/train.
Example Response:
json
Copy
Edit
{
  "message": "Model trained and saved successfully!"
}
4. Predict Downtime
URL: /predict
Method: POST
Description: Predicts downtime based on input features.
Input (JSON):
json
Copy
Edit
{
  "Process temperature [K]": 310,
  "Tool wear [min]": 100
}
Example Request in Postman:
Select POST.
Set the URL to http://127.0.0.1:5000/predict.
Go to the Body tab, select raw, and set the content type to JSON.
Enter the JSON input:
json
Copy
Edit
{
    "Process temperature [K]": 310,
    "Tool wear [min]": 100
}
Example Response:
json
Copy
Edit
{
  "Downtime": "No",
  "Confidence": 0.85
}
Using the API in Postman
Install Postman: Download and install Postman from here.

Add a Request:

Click on New Tab or New Request.
Set the Method and URL:

Use the POST method.
Set the URL to one of the endpoints, e.g., http://127.0.0.1:5000/upload.
For File Upload:

Go to the Body tab and select form-data.
Add a key named file, set it to File, and upload your CSV.
For Prediction:

Go to the Body tab, select raw, and set the content type to JSON.
Enter the JSON data for prediction.
Send the Request:

Click Send and review the response

Future Improvements
Add support for additional machine learning models.
Provide more detailed metrics (e.g., precision, recall).
Deploy the API to a cloud platform like AWS, Azure, or Heroku.









