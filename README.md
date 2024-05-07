# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.



## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:MATHESWARAN K
RegisterNumber:  212222110024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\sem-1\Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop("salary",axis=1)
dataset ["gender"] = dataset ["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset ["hsc_b"].astype('category')
dataset ["degree_t"] = dataset ["degree_t"].astype('category')
dataset ["workex"] = dataset ["workex"].astype('category')
dataset["specialisation"] = dataset ["specialisation"].astype('category')
dataset ["status"] = dataset["status"].astype('category')
dataset ["hsc_s"] = dataset ["hsc_s"].astype('category')
dataset.dtypes
dataset ["gender"] = dataset ["gender"].cat.codes
dataset ["ssc_b"] = dataset["ssc_b"].cat.codes
dataset ["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset ["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset ["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
*/
```

## Output:
### dataset:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/049fe268-a8c9-437f-94ec-adc90883eacb)
### dataset.dtypes:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/f22e40e5-b76f-4177-83a6-3e3483ab92c3)
### dataset:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/26aadedc-88b2-4949-8673-d89dd5941bd3)
### Y:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/723a7ec2-b989-439f-8267-98477dfb4509)
### y_pred:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/ddb71a22-4d78-404e-832b-3c945d1a051c)
### Y:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/a3ebc1ac-32f0-4561-b81f-f72ba20cdd30)
### y_prednew:
![image](https://github.com/mathes6112004/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119477782/a94fe26a-1111-4f47-84ee-56029b544d16)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

