# %% [markdown]
# #CIS 5200: Machine Learning Homework 2

# %%
import os
import sys

# For autograder only, do not modify this cell.
# True for Google Colab, False for autograder
NOTEBOOK = (os.getenv('IS_AUTOGRADER') is None)
if NOTEBOOK:
    print("[INFO, OK] Google Colab.")
else:
    print("[INFO, OK] Autograder.")
    sys.exit()

# %%
# %%capture
!pip install penngrader-client

# %%
%%writefile config.yaml
grader_api_url: 'https://23whrwph9h.execute-api.us-east-1.amazonaws.com/default/Grader23'
grader_api_key: 'flfkE736fA6Z8GxMDJe2q8Kfk8UDqjsG3GVqOFOa'

# %%
from penngrader.grader import PennGrader

# PLEASE ENSURE YOUR PENN-ID IS ENTERED CORRECTLY. IF NOT, THE AUTOGRADER WON'T KNOW WHO
# TO ASSIGN POINTS TO YOU IN OUR BACKEND
STUDENT_ID = 17994725 # YOUR PENN-ID GOES HERE AS AN INTEGER #
SECRET = STUDENT_ID

grader = PennGrader('config.yaml', 'cis5200_sp25_HW2', STUDENT_ID, SECRET)

# %%
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
import cvxpy as cp

from PIL import Image
from dill.source import getsource

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

# %% [markdown]
# # Dataset: Wine Quality Prediction
# 
# Some research on blind wine tasting has suggested that [people cannot taste the difference between ordinary and pricy wine brands](https://phys.org/news/2011-04-expensive-inexpensive-wines.html). Indeed, even experienced tasters may be as consistent as [random numbers](https://www.seattleweekly.com/food/wine-snob-scandal/). Is professional wine tasting in shambles? Maybe ML can take over.
# 
# In this problem set, we will train some simple linear models to predict wine quality. We'll be using the data from [this repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) for both the classification and regression tasks. The following cells will download and set up the data for you.

# %%
%%capture
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names

# %%
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

red_df = pd.read_csv('winequality-red.csv', delimiter=';')

X = torch.from_numpy(red_df.drop(columns=['quality']).to_numpy())
y = torch.from_numpy(red_df['quality'].to_numpy())

# Split data into train/test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalize the data to have zero mean and standard deviation,
# and add bias term
mu, sigma = X_train.mean(0), X_train.std(0)
X_train, X_test = [ torch.cat([((x-mu)/sigma).float(), torch.ones(x.size(0),1)], dim=1)
                    for x in [X_train, X_test]]

# Transform labels to {-1,1} for logistic regression
y_binary_train, y_binary_test = [ (torch.sign(y - 5.5)).long()
                                  for y in [y_train, y_test]]
y_regression_train, y_regression_test = [ y.float() for y in [y_train, y_test]]

# %% [markdown]
# #1. Logistic Regression
# In this first problem, you will implement a logistic regression classifier to classify good wine (y=1) from bad wine (y=0). Your professor has arbitrarily decided that good wine has a score of at least 5.5. The classifier is split into the following components:
# 
# Loss (3pts) & gradient (3pts) - given a batch of examples  
# X
#   and labels  
# y
#   and weights for the logistic regression classifier, compute the batched logistic loss and gradient of the loss with with respect to the model parameters  
# w
#  . Note that this is slightly different from the gradient in Homework 0, which was with respect to the sample  
# X
#  .
# Fit (2pt) - Given a loss function and data, find the weights of an optimal logistic regression model that minimizes the logistic loss
# Predict (3pts) - Given the weights of a logistic regression model and new data, predict the most likely class
# We provide an generic gradient-based optimizer for you which minimizes the logistic loss function, you can call it with LogisticOptimizer().optimize(X,y). It does not need any parameter adjustment.
# 
# Hint: The optimizer will minimize the logistic loss. So this value of this loss should be decreasing over iterations

# %%
def logistic_loss(X, y, w):
    # Given a batch of samples and labels, and the weights of a logistic
    # classifier, compute the batched logistic loss.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
    #
    # w := Tensor(float) of size(d,) --- This is the weights of a logistic
    #     classifer.
    #
    # Return := Tensor of size (m,) --- This is the logistic loss for each
    #     example.

    # Fill in the rest
    return torch.log(1+torch.exp(-y*torch.matmul(X,w)))

def logistic_gradient(X, y, w):
    # Given a batch of samples and labels, compute the batched gradient of
    # the logistic loss.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
    #
    # w := Tensor(float) of size(d,) --- This is the weights of a logistic
    #     classifer.
    #
    # Return := Tensor of size (m,) --- This is the logistic loss for each
    #     example.
    #
    # Hint: A very similar gradient was calculated in Homework 0.
    # However, that was the sample gradient (with respect to X), whereas
    # what we need here is the parameter gradient (with respect to w).

    # Fill in the rest
    output = torch.matmul(X, w)
    return X*(-y * torch.exp(-y * output) / (1 + torch.exp(-y * output))).reshape(-1,1)


def optimize(X, y, niters=100):
    # Given a dataset of examples and labels, minimizes the logistic loss
    # using standard gradient descent.
    #
    # This optimizer is written for you, and you only need to implement the
    # logistic loss and gradient functions above.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
    #
    # Return := Tensor of size(d,) --- This is the fitted weights of a
    #     logistic regression model

    m,d = X.size()
    w = torch.zeros(d)
    print('Optimizing logistic function...')
    for i in range(niters):
        loss = logistic_loss(X,y,w).mean()
        grad = logistic_gradient(X,y,w).mean(0)
        w -= grad
        if i % 50 == 0:
            print(i, loss.item())
    print('Optimizing done.')
    return w

def logistic_fit(X, y):
    # Given a dataset of examples and labels, fit the weights of the logistic
    # regression classifier using the provided loss function and optimizer
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m labels in {-1,1}
    #
    # Return := Tensor of size (d,) --- This is the fitted weights of the
    #     logistic regression model
    import torch

    # Fill in the rest. Hint -- call optimize :-).
    return optimize(X,y)

def logistic_predict(X, w):
    # Given a dataset of examples and fitted weights for a logistic regression
    # classifier, predict the class
    #
    # X := Tensor(float) of size(m,d) --- This is a batch of m examples of
    #    dimension d
    #
    # w := Tensor(float) of size (d,) --- This is the fitted weights of the
    #    logistic regression model
    #
    # Return := Tensor of size (m,) --- This is the predicted classes {-1,1}
    #    for each example
    #
    # Hint: Remember that logistic regression expects a label in {-1,1}, and
    # not {0,1}

    # Fill in the rest
    return torch.sign(torch.matmul(X,w))


# %%
# Test your code on the wine dataset!
# How does your solution compare to a random linear classifier?
# Your solution should get around 75% accuracy on the test set.
torch.manual_seed(42)

d = X_train.size(1)
logistic_weights = {
    'zero': torch.zeros(d),
    'random': torch.randn(d),
    'fitted': logistic_fit(X_train, y_binary_train)
}

for k,w in logistic_weights.items():
    yp_binary_train = logistic_predict(X_train, w)
    acc_train = (yp_binary_train == y_binary_train).float().mean()

    print(f'Train accuracy [{k}]: {acc_train.item():.2f}')

    yp_binary_test = logistic_predict(X_test, w)
    acc_test = (yp_binary_test == y_binary_test).float().mean()

    print(f'Test accuracy [{k}]: {acc_test.item():.2f}')

# %%
grader.grade(test_case_id = 'logistic_loss', answer = getsource(logistic_loss))
grader.grade(test_case_id = 'logistic_gradient', answer = getsource(logistic_gradient))
grader.grade(test_case_id = 'logistic_fit', answer = getsource(logistic_fit))
grader.grade(test_case_id = 'logistic_predict', answer = getsource(logistic_predict))

# %% [markdown]
# # 2. Linear Regression with Ridge Regression
# 
# In this second problem, you'll implement a linear regression model. Similarly to the first problem, implement the following functions:
# 
# 1. Loss (3pts) - Given a batch of examples $X$ and labels $y$, compute the batched mean squared error loss for a linear model with weights $w$.
# 2. Fit (4pts) - Given a batch of examples $X$ and labels $y$, find the weights of the optimal linear regression model
# 3. Predict (3pts) - Given the weights $w$ of a linear regression model and new data $X$, predict the most likely label
# 
# This time, you are not given an optimizer for the fitting function since this problem has an analytic solution. Make sure to test your solution with non-zero ridge regression parameters.
# 
# Hint: You may want to review ridge regression on Slide 22 of Lecture 3.

# %%
def regression_loss(X, y, w):
    # Given a batch of linear regression outputs and true labels, compute
    # the batch of squared error losses. This is *without* the ridge
    # regression penalty.
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(int) of size (m,) --- This is a batch of m real-valued labels
    #
    # w := Tensor(float) of size(d,) --- This is the weights of a linear
    #     classifer
    #
    # Return := Tensor of size (m,) --- This is the squared loss for each
    #     example

    # Fill in the rest
    return torch.pow(y - torch.matmul(X,w), 2)

def regression_fit(X, y, ridge_penalty=1.0):
    # Given a dataset of examples and labels, fit the weights of the linear
    # regression classifier using the provided loss function and optimizer
    #
    # X := Tensor(float) of size (m,d) --- This is a batch of m examples of
    #     of dimension d
    #
    # y := Tensor(float) of size (m,) --- This is a batch of m real-valued
    #     labels
    #
    # ridge_penalty := float --- This is the parameter for ridge regression
    #
    # Return := Tensor of size (d,) --- This is the fitted weights of the
    #     linear regression model
    #
    # Fill in the rest
    inner = torch.matmul(X.t(), X) + ridge_penalty*X.size(0)*torch.eye(X.size(1))
    outer = torch.matmul(X.t(), y)
    return torch.matmul(torch.inverse(inner), outer)

def regression_predict(X, w):
    # Given a dataset of examples and fitted weights for a linear regression
    # classifier, predict the label
    #
    # X := Tensor(float) of size(m,d) --- This is a batch of m examples of
    #    dimension d
    #
    # w := Tensor(float) of size (d,) --- This is the fitted weights of the
    #    linear regression model
    #
    # Return := Tensor of size (m,) --- This is the predicted real-valued labels
    #    for each example
    #
    # Fill in the rest
    return torch.matmul(X,w)

# %%
# Test your code on the wine dataset!
# How does your solution compare to a random linear classifier?
# Your solution should get an average squard error of about 8.6 test set.
torch.manual_seed(42)

d = X_train.size(1)
regression_weights = {
    'zero': torch.zeros(d),
    'random': torch.randn(d),
    'fitted': regression_fit(X_train, y_regression_train)
}

for k,w in regression_weights.items():
    yp_regression_train = regression_predict(X_train, w)
    squared_loss_train = regression_loss(X_train, y_regression_train, w).mean()

    print(f'Train accuracy [{k}]: {squared_loss_train.item():.2f}')

    yp_regression_test = regression_predict(X_test, w)
    squared_loss_test = regression_loss(X_test, y_regression_test, w).mean()

    print(f'Test accuracy [{k}]: {squared_loss_test.item():.2f}')

# %%
grader.grade(test_case_id = 'regression_loss', answer = getsource(regression_loss))
grader.grade(test_case_id = 'regression_fit', answer = getsource(regression_fit))
grader.grade(test_case_id = 'regression_predict', answer = getsource(regression_predict))

# %%



