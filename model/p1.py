#import the libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pickle import dump
import os 

#load the data
data_path = os.path.join(os.getcwd(), "data/data.csv")
data = pd.read_csv(data_path)
print(data)

#check for null data
print(data.isnull().sum())

#features and target
features = data.drop(["StudentID","Gender","Ethnicity","ParentalEducation","GPA"],axis = "columns")
target = data["GPA"]

#training and testing
x_train,x_test,y_train,y_test = train_test_split(features.values , target)

#model creation
model = LinearRegression()
model.fit(x_train,y_train)

trs = model.score(x_train,y_train)
tes = model.score(x_train,y_train)
print("Training Score = ",trs)
print("Testing Score = ",tes)

#pickle file
f = open("gpa.pkl","wb")
dump(model,f)
f.close()





