# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the given dataset.

2. Fitting the dataset into the training set and test set.

3. Applying the feature scaling method.

4. Fitting the logistic regression into the training set.

5. Prediction of the test and result

6. Making the confusion matrix

7. Visualizing the training set results.


## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vinush.cv  
RegisterNumber:212222230176
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:,[0,1]]
y = data[:,2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction(theta,X,Y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T,h-y) / X.shape[0]
  return J,grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J, grad  = costFunction(theta,X_train,y)
print(J)
print(grad)
def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
  x_min ,x_max = X[:,0].min()-1,X[:,0].max()+1
  y_min ,y_max = X[:,1].min()-1,X[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                      np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
plotDecisionBoundary(res.x,X,y)
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/4d67e163-1b78-4a46-9f38-ca2ae16c6e8b)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/79201558-27ff-47c1-b0ff-74bc15ad524c)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/87e11a59-5ce3-42a5-95c7-8c0b5f1f5d85)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/4e7a9234-6cc3-46ff-942c-c9f7f863b138)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/f263e5ec-f782-4623-b328-78f37d3c2f8e)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/bb9647ff-33d3-4837-b46b-29976756a7c3)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/c386ed0c-0807-4bf7-b78a-7d43e9b85805)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/a74f20d3-df2c-4747-b5f2-dc93f155454d)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/6ea3c5c6-f547-4aac-a383-6173457879d3)



![image](https://github.com/vinushcv/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113975318/0f6cd65e-f327-44ac-8108-c5653c674a59)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

