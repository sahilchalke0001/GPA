#import the libraries
from pickle import load
import os

# Load the updated model
model_path = os.path.join(os.getcwd(), "gpa_updated.pkl")
with open(model_path, "rb") as f:
    model = load(f)

# Example input data for prediction
input_data = [[18, 8.19, 0, 0, 1, 1, 0, 0, 0, 3.0, 88, 91, 50, 67]]  

# Prediction
prediction = model.predict(input_data)
print(f"Predicted Overall GPA: {prediction[0]}")
