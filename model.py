import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib  # To save and load the model
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.trained = False

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        Assumes the CSV has columns: Machine_ID, Temperature, Run_Time, Downtime_Flag
        """
        self.data = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return self.data

    def preprocess_data(self):
        """
        Preprocess the data for training.
        - Features: Temperature, Run_Time
        - Target: Downtime_Flag
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Features and target
        self.X = self.data[['Process temperature [K]', 'Tool wear [min]']]
        self.y = self.data['Target']

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Data preprocessed and split into training and testing sets.")

    def train(self):
        """
        Train the Logistic Regression model.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data available. Please preprocess data first.")

        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print("Model trained successfully!")

        # Evaluate the model
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

    def predict(self, input_data):
        """
        Make predictions using the trained model.
        :param input_data: A dictionary with keys 'Temperature' and 'Run_Time'.
        :return: Prediction (0 or 1) and confidence score.
        """
        if not self.trained:
            raise ValueError("Model is not trained. Please train the model first.")

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)[0]
        confidence = self.model.predict_proba(input_df)[0][1]  # Confidence for class 1 (Downtime: Yes)

        return prediction, confidence

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        """
        if not self.trained:
            raise ValueError("Model is not trained. Please train the model first.")

        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load a trained model from a file.
        """
        self.model = joblib.load(file_path)
        self.trained = True
        print(f"Model loaded from {file_path}")

    def plot_decision_boundary(self):
        """
        Plot the data points and the decision boundary of the Logistic Regression model.
        """
        if not self.trained:
            raise ValueError("Model is not trained. Please train the model first.")

        # Create a mesh grid for plotting the decision boundary
        x_min, x_max = self.X['Process temperature [K]'].min() - 1, self.X['Process temperature [K]'].max() + 1
        y_min, y_max = self.X['Tool wear [min]'].min() - 1, self.X['Tool wear [min]'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # Predict for each point in the mesh grid
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

        # Plot the data points
        plt.scatter(self.X['Process temperature [K]'], self.X['Tool wear [min]'], c=self.y, cmap='coolwarm', edgecolors='k', marker='o')
        plt.xlabel('Process temperature [K]')
        plt.ylabel('Run Time')
        plt.title('Logistic Regression Decision Boundary')
        plt.colorbar(label='Downtime Flag (0: No, 1: Yes)')
        plt.show()