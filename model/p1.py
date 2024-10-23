#import the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pickle import dump
import os

# Load the data
data_path = os.path.join(os.getcwd(), "data/data_updated.csv")
data = pd.read_csv(data_path)
print(data.head())  

# Check for missing values
print(data.isnull().sum())


features = data.drop(["StudentID", "Gender", "Ethnicity", "ParentalEducation", "Overall GPA", "OS GPA", "DBMS GPA", "ML GPA", "Python GPA", "GPA"], axis="columns")
target = data["Overall GPA"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features.values, target, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Evaluate the model on training and testing sets
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f"Training Score: {train_score}")
print(f"Testing Score: {test_score}")

# Save the trained model to a pickle file
with open("gpa_updated.pkl", "wb") as f:
    dump(model, f)

print("Model has been saved as 'gpa_updated.pkl'")






