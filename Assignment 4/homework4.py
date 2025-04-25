# %% [markdown]
# # CIS 5200: Machine Learning
# ## Homework 4

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

# %% [markdown]
# ### Penngrader setup

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

grader = PennGrader('config.yaml', 'cis5200_sp25_HW4', STUDENT_ID, SECRET)

# %%
# packages for homework
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

from inspect import getsource

# %% [markdown]
# # 1. Boosting (5 pts)
# 
# In this problem, you'll implement a basic boosting algorithm on the binary classification breast cancer dataset. Here, we've provided the following weak learner for you: an $\ell_2$ regularized logistic classifier trained with gradient descent.

# %%
class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.linear = nn.Linear(30,1)
    def forward(self, X):
        out = self.linear(X)
        return out.squeeze()

def fit_logistic_clf(X,y):
    clf = Logistic()
    opt = torch.optim.Adam(clf.parameters(), lr=0.1, weight_decay=1e2)
    loss = torch.nn.BCEWithLogitsLoss()
    for t in range(200):
        out = clf(X)
        opt.zero_grad()
        loss(out,(y>0).float()).backward()
        # if t % 50 == 0:
        #     print(loss(out,y.float()).item())
        opt.step()
    return clf

def predict_logistic_clf(X, clf):
    return torch.sign(clf(X)).squeeze()

# %% [markdown]
# Your task is to boost this logistic classifier to reduce its bias. Implement the following two functions:
# 
# + Finish the boosting algorithm: we've provided a template for the boosting algorithm in `boosting_fit`, however it is missing several components. Fill in the missing snippets of code.
# + Prediction after boosting (2pts): implement `boosting_predict` to make predictions with a given boosted model.

# %%
def boosting_fit(X, y, T, fit_logistic_clf, predict_logistic_clf):
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
    # T := Maximum number of models to be implemented

    m = X.size(0)
    clfs = []
    mu = torch.ones(m) / m

    while len(clfs) < T:
        # Calculate the weights for each sample mu. You may need to
        # divide this into the base case and the inductive case.

        ## ANSWER
        mu /= mu.sum()
        ## END ANSWER

        # Here, we draw samples according to mu and fit a weak classifier
        idx = torch.multinomial(mu, m, replacement=True)
        X0, y0 = X[idx], y[idx]

        clf = fit_logistic_clf(X0, y0)

        # Calculate the epsilon error term

        ## ANSWER
        prediction = predict_logistic_clf(X, clf)
        eps = torch.sum(mu * (prediction != y))
        ## END ANSWER

        if eps > 0.5:
            # In the unlikely even that gradient descent fails to
            # find a good classifier, we'll skip this one and try again
            continue

        # Calculate the alpha term here

        ## ANSWER
        alpha = torch.log2((1 - eps) / eps) / 2
        mu *= torch.exp(-alpha * y * prediction)
        mu /= mu.sum()
        ## END ANSWER

        clfs.append((alpha,clf))
    return clfs

def boosting_predict(X, clfs, predict_logistic_clf):
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # clfs := list of tuples of (float, logistic classifier) -- the list of boosted classifiers
    # Return := Tnesor(int) of size (m) -- the predicted labels of the dataset

    output = torch.zeros(X.size(0))
    for alpha, clf in clfs:
        output += alpha * predict_logistic_clf(X, clf)
    return torch.sign(output).squeeze()


# %% [markdown]
# Test out your code on the breast cancer dataset. As a sanity check, your statndard logistic classifier will get a train/test accuracy of around 80% while the boosted logistic classifier will get a train/test accuracy of around 90%.

# %%
from sklearn.datasets import load_breast_cancer
cancer = datasets.load_breast_cancer()
data=train_test_split(cancer.data,cancer.target,test_size=0.2,random_state=123)

torch.manual_seed(123)

X,X_te,y,y_te = [torch.from_numpy(A) for A in data]
X,X_te,y,y_te = X.float(), X_te.float(), torch.sign(y.long()-0.5), torch.sign(y_te.long()-0.5)


logistic_clf = fit_logistic_clf(X,y)
print("Logistic classifier accuracy:")
print('Train accuracy: ', (predict_logistic_clf(X, logistic_clf) == y).float().mean().item())
print('Test accuracy: ', (predict_logistic_clf(X_te, logistic_clf) == y_te).float().mean().item())

boosting_clfs = boosting_fit(X,y, 10, fit_logistic_clf, predict_logistic_clf)
print("Boosted logistic classifier accuracy:")
print('Train accuracy: ', (boosting_predict(X, boosting_clfs, predict_logistic_clf) == y).float().mean().item())
print('Test accuracy: ', (boosting_predict(X_te, boosting_clfs, predict_logistic_clf) == y_te).float().mean().item())

# %% [markdown]
# ## Autograder

# %%
grader.grade(test_case_id = 'boosting_fit', answer = getsource(boosting_fit))
grader.grade(test_case_id = 'boosting_predict', answer = getsource(boosting_predict))

# %% [markdown]
# # 2. Extending Reverse Mode Auto-differentiation (8 pts)
# 
# In lecture we learned about how auto-differentiation could be used to automatically calculate gradients through a computational graph. Crucially, all we needed to do was know how to propagate variations of the chain rule at each individual node in the graph.
# 
# The PyTorch framework makes it very simple to extend auto-differentation to new formula or expressions. Specifically, PyTorch keeps track of the computational graph with the `forward` function, saving relevant computations, and then computes the chain rule with the `backward` function (in reverse mode).
# 
# For example, consider the Legendre polynomial of degree 3, $P_3(x) = \frac{1}{2}(5x^3 - 3x)$. How would we implement our own custom module to do this, if it didn't already exist in PyTorch? We can do this like in the following ([example taken from the PyTorch documetnation](https://pytorch.org/docs/stable/notes/extending.html)):  

# %%
class LegendrePolynomial3(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

# %% [markdown]
# And that's it! We can now use the `LegendrePolynomial3` function in combination with any other PyTorch functions and automatically calculate gradients with auto-differentiation. Note that, since PyTorch uses *reverse-mode*, in the `backward` function we are:  
# 1. Given the gradient of the loss with respect to the output of the function, and
# 2. Need to recompute the gradient of the loss with respect to the inputs of the function.
# 
# This is an extremely general and powerful framework. For example, researchers have implemented forward functions and their gradients for modules as complex as a call to an external simulator such as a physics engine, which would then let you differentiate through the simulator!
# 
# In this exercise, you'll implement a new PyTorch function for a simpler function, the integral:
# $$f(x) = \int_{-\infty}^x g(t)dt$$
# PyTorch doesn't have a function for doing numerical integration, so we'll need to implement the backwards function ourselves if we want to use auto-differentiate with integrals. In particular, we'll do this for the following piece-wise constant function:
# 
# $$g(x; \beta, \eta) = \sum_{r=1}^R \beta_r \mathbb 1[x\in (\eta_{r-1},\eta_r)]$$
# 
# for parameters $\beta\in \mathbb R^R$ and $\eta\in \mathbb R^{R+1}$.
# 
# 1. Forward (2pts): Implement the `forward` function which calculates $f(x)$ for a batch of example $x$.
# 2. Backward (6pts): Implement the `backward` function which calculates the reverse-mode auto-differentiation rule for $f$ with respect to $x,\beta, \eta$, 2 points each.
# 
# Hints:
# + Recall that $g(x)$ is a piece-wise constant function, and the integral of a function is the area between the function and 0. Thus, you can reformulate the integral of each constant part of $g$ as simply the (signed) area of the rectangle, and so the integral of $g(x)$ is the sum of the areas of all the piece-wise rectangles up to $x$
# + Recall that for reverse-mode autodifferentiation, we are given the derivative of the loss with respect to the output, or $\frac{\partial \ell(x))}{\partial f(x; \beta, \eta)}$, and our goal is to then compute $\frac{\partial \ell(x))}{\partial x}, \frac{\partial \ell(x))}{\partial \beta}, \frac{\partial \ell(x))}{\partial \eta}$.
# + Remember the chain rule: $$\frac{\partial \ell(x))}{\partial x} = \sum_i \frac{\partial \ell(x))}{\partial f(x; \beta, \eta)_i} \cdot \frac{\partial f(x; \beta, \eta))_i}{\partial x}$$
# + For simplicity, you can assume that all $x$'s and $\eta$'s are distinct so you don't have to worry about border cases for the intervals.

# %%
class IntegralPiecewiseConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, beta, eta):
        # ctx := A PyTorch object for saving data for use in the backward
        #   function
        # X := Tensor(float) of size (m) -- a minibatch of examples
        # beta := Tensor(float) of size (k) -- the magnitudes of each part of
        #   the piece-wise constant function
        # eta := Tensor(float) of size (k+1) -- the start/end points of each
        #   part of the piece-wise constant function
        # Return the integral of the piece-wise constant function applied to
        #   each entry of  X

        # Here we save the inputs to the forward function to use in backward
        ctx.save_for_backward(X, beta, eta)

        # Fill in the rest
        m = X.size(0)
        n = beta.size(0)
        ans = torch.zeros(m)

        for i in range(m):
            x = X[i]
            xi_int = 0
            for j in range(n):
                if x > eta[j]:
                    xi_int += beta[j] * (min(x, eta[j+1]) - eta[j])
            ans[i] = xi_int

        return ans

    @staticmethod
    def backward(ctx, grad_output):
        # ctx := A PyTorch object for containing saved data from the forward
        #   function
        # grad_output := Tensor(float) of size (m) -- a minibatch of gradients
        #   of the loss with respect to the output of this function. Since we
        #   are working with a scalar function, each element is the gradient
        #   with respect to an output of the minibatch. In other words, this is
        #   dL/df(x) where this module outputs f(x).
        # Return a tuple containing the gradient of the loss with respect to
        #   the inputs X, beta, and eta: (dL/dX, dL/dbeta, dL/deta).

        # Here we retrieve the tensors from the forward pass
        X, beta, eta = ctx.saved_tensors
        print(grad_output)

        # Fill in the rest
        m = X.size(0)
        n = beta.size(0)
        p = eta.size(0)

        dX = torch.zeros(m)
        dbeta = torch.zeros(n)
        deta = torch.zeros(p)

        for i in range(m):
            x  = X[i]
            gradient = grad_output[i]
            for j in range(n):
                if eta[j] < x and x <= eta[j + 1]:
                    dX[i] += gradient * beta[j]
            for j in range(n):
                if x > eta[j]:
                    dbeta[j] += gradient * (min(x, eta[j + 1]) - eta[j])
            for j in range(n):
                if x > eta[j]:
                    deta[j] -= beta[j] * gradient
                if x > eta[j + 1]:
                    deta[j + 1] += beta[j] * gradient

        return dX, dbeta, deta    

# %% [markdown]
# As an example, calculating the gradients for the following example will result in the following output:
# ```
# tensor([ 1., -1.,  3.])
# tensor([0.2500, 0.4000, 0.7000, 0.1000])
# tensor([-3., -2.,  6., -4.,  0.])
# ```

# %%
X = torch.Tensor([0.05,0.5,0.9])
betas = torch.Tensor([1,2,-1,3])
etas = torch.Tensor([0,0.1,0.3,0.8,1])

X.requires_grad = True
betas.requires_grad = True
etas.requires_grad = True
IntegralPiecewiseConstant.apply(X, betas, etas).sum().backward()

print(X.grad)
print(betas.grad)
print(etas.grad)

# %%
IPC_str = getsource(IntegralPiecewiseConstant.forward) + getsource(IntegralPiecewiseConstant.backward)
grader.grade(test_case_id = 'autodiff_forward', answer = IPC_str)
grader.grade(test_case_id = 'autodiff_backward', answer = IPC_str)

# %% [markdown]
# # 3. Neural Networks and Gradient Descent (5 pts)
# 
# In the previous example, we directly calculated the gradient of a function with respect to various inputs using PyTorch's `autograd` library. As we did in that problem, one can use this autograd library to directly implement gradient descent by iterating over all parameters and applying the gradient update. However, as the number of parameters grow, directly implementing these updates can become quite onerous. To handle neural networks with lots of parameters, the PyTorch library includes optimizers that make training with gradient descent very easy with the following 5 steps:
# 1. Create an optimizer object and give it all the parameters you'd like to optimize
# 2. Calculate a loss that you'd like to minimize
# 3. Clear old gradients
# 4. Calculate new gradients
# 5. Update the parameters with one gradient step
# 
# The end result is a generic boilerplate recipe that will optimize *any* objective with gradient descent. Here is an example running gradient descent on a linear model, using the stochastic gradient descent (SGD) optimizer and a basic dataloader.

# %%
# Setup a simple problem
m,d = 128,5
X = torch.randn(m,d)
w_opt, b_opt = torch.randn(d), torch.randn(1)
y = X.matmul(w_opt) + b_opt

# setup the dataloader
simple_dataset = torch.utils.data.TensorDataset(X,y)
loader = torch.utils.data.DataLoader(simple_dataset,batch_size=16)

# Create the model
lin = nn.Linear(d,1)

# setup the optimizer (Step 1)
opt = torch.optim.SGD(lin.parameters(), lr=0.001)

# iterate over epochs
for i in range(100):
    # iterate over minibatches
    for X0,y0 in loader:
        yhat = lin(X0).squeeze(1) # make predictions
        loss = F.mse_loss(yhat,y0) # calculate loss (Step 2)

        opt.zero_grad() # clear gradients from previous iteration (Step 3)
        loss.backward() # calculate new gradients (Step 4)
        opt.step() # update parameters (Step 5)

    # logging
    with torch.no_grad():
        if i % 10 == 0:
            print(loss.item())

# %% [markdown]
# 
# 
# In the second part of this assignment, we'll implement a basic neural network for a tree cover classification problem using the PyTorch library. Here, the problem is to use 12 features (which have been expanded to 54 columns of data to expand categorical variables into binary features) to predict one of 7 tree cover types. A full description of the dataset can be found [here](http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info). The following cell will download the data and convert it into PyTorch tensors for you, with a training split of 2000 examples per class and a validation split of 500 examples per class.

# %%
!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# fetch dataset
covertype = fetch_ucirepo(id=31)

# data (as pandas dataframes)
X = covertype.data.features
y = covertype.data.targets

# metadata
# print(covertype.metadata)

# variable information
# print(covertype.variables)

# %%
df = pd.concat([covertype.data.features, covertype.data.targets], axis = 1)
df = pd.concat([df.iloc[df.values[:,-1]==i].sample(2500) for i in range(1,8)])
X,y = df.values[:,:-1], df.values[:,-1]-1 # re-index labels to 0-6
X,y = [torch.from_numpy(a) for a in (X,y)] # convert to PyTorch

dataset = torch.utils.data.TensorDataset(X,y)
train_set,val_set = torch.utils.data.random_split(dataset,[2000*7,500*7]) # generate splits

# %% [markdown]
# Your goal is to achieve at least 70% accuracy on this forest cover task. We suggest a very simple neural network.
# 
# 1. (5pts) Implement a model for predicting forest cover. You will get 1 point for every first 14% accuracy, up to 70% for a total of 5 points. This is achievable with a small neural network with one hidden layer and ReLU activations. It is possible to achieve over 80% accuracy with less than 15 seconds (many solutions exist that take less than a minute).
# 
# It is possible to get much higher than 75%, in fact it is possible to exceed 80% accuracy. **You do not need a GPU**.
# 
# Hints:
# + A simple neural network directly on the data can get around 50% accuracy. You can start with a sequential model, linear layers, and ReLU activations (PyTorch has a wide range of modules [here](https://pytorch.org/docs/stable/nn.html)).
# + To iterate over minibatches, PyTorch has a useful `DataLoader` object with documentation [here](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader).
# + Data normalization can be a very important factor, not just in deep learning. Check your dataset statistics and normalize to improve your accuracy further.
# + You may need to explore different types of optimizers and learning rates.  
# + Remember to check your performance on the validation set, and not the training set. You will need to perform well on the held-out test data.
# + To help with your testing, you can add additional parameters via keyword arguments to tweak your pipeline.
# + There is no single right answer here---the only goal is to get to 70%!
# + Solutions that exploit the grader to extract test set labels will receive no credit.

# %%
class ForestCover():
    def __init__(self, train_set, nhidden=100):
        # train_set := a PyTorch dataset of training examples from the tree
        #   cover prediction problem.

        # Initialize your model here!
        self.model = nn.Sequential(
            nn.Linear(train_set[0][0].shape[0], nhidden),
            nn.ReLU(),
             nn.Linear(nhidden, int(nhidden/2)),
            nn.ReLU(),
            nn.Linear(int(nhidden/2), 7),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = torch.nn.CrossEntropyLoss()

        inputs = torch.stack([x for x, _ in train_set])
        inputs = inputs.float()
        self.mean = inputs.mean(dim=0)
        self.std = inputs.std(dim=0)

    def train(self, train_set, epochs=20):
        # train_set := a PyTorch dataset of training examples from the tree
        #   cover prediction problem.

        # Train your model here!
        train_data = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            for X, y in train_data:
                std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
                X = (X.float() - self.mean) / std                 

                output = self.model(X)
                batch_loss = self.loss_function(output, y.long())

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()


    def predict(self, X):
        # X := a Tensor(float) of size (m,d) of examples for the model to
        #   to predict.

        # Make predictions on a new input here!
        self.model.eval()
        with torch.no_grad():
            X = (X.float() - self.mean) / torch.where(self.std == 0, torch.ones_like(self.std), self.std)
            output = self.model(X)
            prediction = torch.argmax(output, dim=1)
        return prediction

solution = ForestCover(train_set)
solution.train(train_set)

acc = []
for X,y in torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False):
    y_pred = solution.predict(X)
    acc.append(y_pred == y)

acc = torch.cat(acc).float().mean()
print(f"Validation accuracy: {acc:.2f}")

# %% [markdown]
# Your code will be scored according to accuracy on a held-out test set.
# If you don't use your validation set during training, your performance on the validation set will be approximately your performance on the test set.
# **Warning: solutions that exploit the grader to extract test set labels will receive a manual adjustment for zero credit.**

# %%
!wget https://machine-learning-upenn.github.io/assets/hw3/X_test.pth -O "X_test.pth"
X_test = torch.load("X_test.pth")

# %%
y_soln = solution.predict(X_test)
grader.grade(test_case_id = 'forestcover', answer = y_soln.__str__())

# %%



