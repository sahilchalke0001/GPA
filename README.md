
# GPA Prediction Using Linear Regression

## Table of Contents
- [1. Understanding Linear Regression](#1-understanding-linear-regression)
- [2. Data Collection](#2-data-collection)
- [3. Data Preparation](#3-data-preparation)

## 1. Understanding Linear Regression

Linear regression is a supervised machine learning algorithm that finds the best-fit line (linear relationship) between independent variables (features) and a dependent variable (target). The formula for a simple linear regression model is:

\[
y = mx + b
\]

Where:
- \(y\) is the dependent variable (GPA).
- \(x\) is the independent variable (e.g., study hours).
- \(m\) is the slope of the line (the coefficient).
- \(b\) is the y-intercept.

For multiple linear regression, the equation expands to include multiple independent variables:

\[
y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
\]

Where:
- \(b_0\) is the intercept.
- \(b_1, b_2, ..., b_n\) are the coefficients for each independent variable \(x_1, x_2, ..., x_n\).

## 2. Data Collection

To predict GPA, you'll first need to gather a dataset that includes:

- **Dependent Variable:** The GPA scores of students (usually on a scale from 0.0 to 4.0).
- **Independent Variables:** These could include:
  - Study hours per week
  - Attendance percentage
  - Extracurricular activities
  - Class participation
  - Parent Support
  - Any other relevant factors

## 3. Data Preparation

1. **Cleaning the Data:** Handle missing values, outliers, and erroneous entries.
2. **Feature Selection:** Identify which independent variables are relevant to predicting GPA.

3. **Splitting the Data:** Divide your dataset into training and testing sets (typically 70-80% for training and 20-30% for testing).




## Acknowledgements

We would like to extend our sincere thanks to [Ruturaj Sanap](https://github.com/Ruturaj1007) for his invaluable contribution to building the multiple linear regression model for this project. His efforts in analyzing the data, selecting the appropriate features, and applying statistical methods to create a robust model were instrumental in the success of this project.

Through his work, we were able to develop a predictive model that accurately captures the relationships between multiple independent variables and the target variable, significantly improving our ability to make reliable predictions. His deep understanding of regression analysis and data preprocessing helped in delivering a model that is both efficient and effective.


## Deployment

The project has been successfully deployed on Render, a popular cloud hosting platform known for its simplicity and ability to deploy web applications seamlessly. The deployment process involved hosting a Streamlit application, which is a Python framework designed for creating data-driven applications with ease.

```bash
 streamlit run frontend/app.py
```


## 🔗 Links
[![GPA Predictor](https://img.shields.io/badge/GPA_Predictor-000?style=for-the-badge&logo=github&logoColor=white)](https://gpa-ltaw.onrender.com/)


## Screenshots

[![Screenshot-2024-10-14-235849.png](https://i.postimg.cc/QCN5LJP9/Screenshot-2024-10-14-235849.png)](https://postimg.cc/gX5r88Gm)

