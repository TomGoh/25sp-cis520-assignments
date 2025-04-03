# %% [markdown]
# # CIS 5200: Machine Learning
# ## Homework 3

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
# packages for homework
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn import datasets
from sklearn.model_selection import train_test_split

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

# %%
def get_class_source(cls):
    import re
    class_name = cls.__name__
    from IPython import get_ipython
    ipython = get_ipython()
    inputs = ipython.user_ns['In']
    pattern = re.compile(r'^\s*class\s+{}\b'.format(class_name))
    for cell in reversed(inputs):
        if pattern.search(cell):
            return cell
    return None

# %%
# Autograder will be announced on Ed Discussion approximately a week after initial release
from penngrader.grader import PennGrader
from dill.source import getsource

# PLEASE ENSURE YOUR PENN-ID IS ENTERED CORRECTLY. IF NOT, THE AUTOGRADER WON'T KNOW WHO
# TO ASSIGN POINTS TO YOU IN OUR BACKEND
STUDENT_ID = 17994725 # YOUR PENN-ID GOES HERE AS AN INTEGER #
SECRET = STUDENT_ID

grader = PennGrader('config.yaml', 'cis5200_sp25_HW3', STUDENT_ID, SECRET)

# %% [markdown]
# ### 0. Gradients with PyTorch
# 
# At this point, you've implemented a lot of gradients. However, these days, manual implementation of gradients is a thing of the past: PyTorch is a scientific computing library that comes with the ability to automatically compute gradients for you! This is called auto-differentiation. Here is an example of using auto-differentiation to compute the gradient of a quadratic function, $f(x) = ax^2$. The key parts are as follows:
# 
# 1. Variables that you want to differentiate with respect to should have the `requires_grad` flag set to `True`.
# 2. Calculate the objective that you'd like the compute the gradient of, using the variable from step (1).
# 3. Pass the objective and the variable you are differentiating to `torch.autograd.grad`.

# %%
# Step 1: Set requires_grad to true for x
x = torch.Tensor([3.0])
a = torch.Tensor([1.5])
x.requires_grad = True

# Step 2: Compute the objective
y = a*(x**2)

# Step 3: Use autograd
grad = torch.autograd.grad([y],[x])[0]

print("PyTorch gradient:", grad)
print("Analytic gradient:", 2*a*x)

# %% [markdown]
# You'll notice that the gradient computed with PyTorch matches exactly the analytic gradient $\nabla f(x) = 2ax$, but without having to implement or derive the analytic gradient! This works for gradients with respect to any sized variables. For example, if $x$ is now a vector, and the objective is $f(x) = a\|x\|_2^2$ then we can calculate the gradient in the same way:

# %%
# Step 1: Set requires_grad to true for x
x = torch.Tensor([3.0, 2.0])
a = torch.Tensor([1.5])
x.requires_grad = True

# Step 2: Compute the objective
y = a*(x.norm(p=2)**2)

# Step 3: Use autograd
grad = torch.autograd.grad([y],[x])[0]

print("PyTorch gradient:", grad)
print("Analytic gradient:", 2*a*x)

# %% [markdown]
# From now on, we highly recommend that you use auto-differentiation to calculate gradients. As long as all of your operations are differentiable, the final objective will be differentiable.
# 

# %% [markdown]
# # 1. SVM and Gradient Descent
# 
# In this first problem, you'll implement (soft margin) support vector machines with gradient descent, using gradients from PyTorch's autodifferentiation library.
# + (2pts) Calculate the objective of the Soft SVM
# + (2pts) Calculate the gradient of the Soft SVM objective
# + (2pts) Implement a basic gradient descent optimizer. Your solution needs to converge to an accurate enough answer.
# + (1pts) Make predictions with the Soft SVM
# 
# Tips:
# - This assignment is more freeform than previous ones. You're allowed to initialize the parameters of the SVM model however you want, as long as your implemented functions return the right values.
# - We recommend using PyTorch's `torch.autograd.grad` to get the gradient instead of deriving the SVM gradient.
# - You'll need to play with the values of step size and number of iterations to
# converge to a good value.
# - To debug your optimization, print the objective over iterations. Remember that the theory says as long as the learning rate is small enough, for strongly convex problems, we are guaranteed to converge at a certain rate. What does this imply about your solution if it is not converging?
# - As a sanity check, you can get around 97.5% prediction accuracy and converge to an objective below 0.16.  

# %%
class SoftSVM():
    def __init__(self, ndims):
        # Here, we initialize the parameters of your soft-SVM model for binary
        # classification. You can change the initialization but don't change
        # the weight and bias variables as the autograder will assume that
        # these exist.
        # ndims := integer -- number of dimensions
        # no return type

        self.weight = torch.randn(ndims)
        self.bias = torch.randn(1)
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def objective(self, X, y, l2_reg):
        # Calculate the objective of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty
        # Returns a scalar tensor (zero dimensional tensor) -- the loss for the model
        # Fill in the rest
        predicted = torch.matmul(X, self.weight) + self.bias
        hinge_loss = torch.mean(torch.max(torch.zeros_like(y), 1 - y*predicted))
        l2_loss = l2_reg*torch.sum(self.weight**2)
        return hinge_loss + l2_loss


    def gradient(self, X, y, l2_reg):
        # Calculate the gradient of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty
        # Return Tuple (Tensor, Tensor) -- the tensors corresponds to the
        # gradients of the weight and bias parameters respectively
        # Fill in the rest
        objective = self.objective(X, y, l2_reg)
        grad = torch.autograd.grad(objective, [self.weight, self.bias])
        weight_grad = grad[0]
        bias_grad = grad[1]
        return weight_grad, bias_grad


    def optimize(self, X, y, l2_reg, learning_rate=0.01):
        # Calculate the gradient of your soft-SVM model
        # X := Tensor of size (m,d) -- the input features of m examples with d dimensions
        # y := Tensor of size (m) -- the labels for each example in X
        # l2_reg := float -- L2 regularization penalty

        # no return type

        # Fill in the rest
        max_iterations = 1000
        threshold = 1e-5
        
        prev_objective = float('inf')
        
        for i in range(max_iterations):
            # Compute current objective
            objective = self.objective(X, y, l2_reg).item()
            
            # Check for convergence
            if abs(prev_objective - objective) < threshold:
                break
                
            # Compute gradients
            weight_grad, bias_grad = self.gradient(X, y, l2_reg)
            
            # Update parameters in-place
            with torch.no_grad():
                self.weight.data -= learning_rate * weight_grad
                self.bias.data -= learning_rate * bias_grad
                
            prev_objective = objective
        

    def predict(self, X):
        # Given an X, make a prediction with the SVM
        # X := Tensor of size (m,d) -- features of m examples with d dimensions
        # Return a tensor of size (m) -- the prediction labels on the dataset X

        # Fill in the rest
        return torch.sign(torch.matmul(X, self.weight) + self.bias)

# %% [markdown]
# Test the Soft SVM on the breast cancer dataset:

# %%
#Load dataset
cancer = datasets.load_breast_cancer()
X,y = torch.from_numpy(cancer['data']), torch.from_numpy(cancer['target'])
mu,sigma = X.mean(0,keepdim=True), X.std(0,keepdim=True)
X,y = ((X-mu)/sigma).float(),(y - 0.5).sign() # prepare data
l2_reg = 0.1
print(X.size(), y.size())

# Optimize the soft-SVM with gradient descent
clf = SoftSVM(X.size(1))
clf.optimize(X,y,l2_reg)
print("\nSoft SVM objective: ")
print(clf.objective(X,y,l2_reg).item())
print("\nSoft SVM accuracy: ")
(clf.predict(X) == y).float().mean().item()

# %%
grader.grade(test_case_id = 'SVM_objective', answer = get_class_source(SoftSVM))
grader.grade(test_case_id = 'SVM_gradient', answer = get_class_source(SoftSVM))
grader.grade(test_case_id = 'SVM_optimize', answer = get_class_source(SoftSVM))
grader.grade(test_case_id = 'SVM_predict', answer = get_class_source(SoftSVM))

# %% [markdown]
# ### 2. Decision Trees and Bagging
# 
# In this problem, we'll implement a simplified version of random forests. We'll be using the `iris` dataset, which has 4 features that are discretized to $0.1$ steps between $0$ and $8$ (i.e. the set of all possible features is $\{0.1, 0.2, \dots, 7.8, 7.9\}$. Thus, all thresholds that we'll need to consider are $\{0.15, 0.25, \dots, 7.75, 7.85\}$.
# 
# Your task in this first part is to finish the implementation of decision trees. We've provided a template for some of the decision tree code, following which you'll finish the bagging algorithm to get a random forest.
# 
# 1. Entropy (2pts): calculate the entropy of a given vector of labels in the `entropy` function. Note that the generalization of entropy to 3 classes is $H = -\sum_{i=1}^3 p_i\log(p_i)$ where $p_i$ is the proportion of examples with label $i$.
# 2. Find the best split (1pt): finish the `find_split` function by finding the feature index and value that results in the split minimizing entropy.
# 3. Build the tree (2pts): finish the `expand_node` function by completing the recursive call for building the decision tree.
# 4. Predict with tree (2pts): implement the `predict_one` function, which makes a prediction for a single example.
# 
# Throughout these problems, the way we represent the decision tree is by using python dicts. In particular, a node is a `dict` that can have the following keys:
# 
# 1. `node['label']` Return the label that we should predict upon reaching this node. node should **only** have a `label` entry if it is a leaf node.
# 2. `node['left']` points to another node dict that represents this node's left child.
# 3. `node['right']` points to another node dict that represents this node's right child.
# 4. `node['split']` is a tuple containing the feature index and value that this node splits left versus right on.
# 
# In our implementation, all comparisons will be **greater than** comparisons, and "yes" answers go **left**. In other words, if `node['split'] = (2, 1.25)`, then we expect to find all data remaining at this node with feature 2 value **greater** than 1.25 in `node['left']`, and feature value **less** than 1.25 in `node['right']`.
# 
# Tips:
# + If you have NaNs, you may be dividing by zero.
# 

# %%
class DecisionTree:
    def entropy(self, y):
        # Calculate the entropy of a given vector of labels in the `entropy` function.
        #
        # y := Tensor(int) of size (m) -- the given vector of labels
        # Return a float that is your calculated entropy

        # Fill in the rest
        classes, count = y.unique(return_counts=True)
        p = count.float()/y.size(0)
        return -torch.sum(p*torch.log2(p))

    def find_split(self, node, X, y, k=4):
        # Find the best split over all possible splits that minimizes entropy.
        #
        # node := Map(string: value) -- the tree represented as a Map, the key will take four
        #   different string: 'label', 'split','left','right' (See implementation below)
        #   'label': a label node with value as the mode of the labels
        #   'split': the best split node with value as a tuple of feature id and threshold
        #   'left','right': the left and right branch with value as the label node
        # X := Tensor of size (m,d) -- Batch of m examples of demension d
        # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
        # k := int -- the number of classes, with default value as 4
        # Return := tuple of (int, float) -- the feature id and threshold of the best split

        m = y.size(0)
        best_H, best_split = 999, None
        features = torch.randint(0, 4,(k,))

        for feature_idx in features:
            for threshold in torch.arange(0.15,7.9,0.1):
                idx = X[:,feature_idx] > threshold
                ####################################
                # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
                ####################################
                if idx.sum() == 0 or idx.sum() == idx.size(0):
                    continue

                m_left = idx.sum()
                m_right = (~idx).sum()

                H_left = self.entropy(y[idx])
                H_right = self.entropy(y[~idx])
                ## ANSWER
                split_H = (m_left/m)*H_left + (m_right/m)*H_right
                ## END ANSWER

                ####################################
                # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
                ####################################
                if split_H < best_H or best_split == None:
                    best_H, best_split = split_H, (feature_idx, threshold)
        return best_split

    def expand_node(self, node, X, y, max_depth=0, k=4):
        # Completing the recursive call for building the decision tree
        # node := Map(string: value) -- the tree represented as a Map, the key will take four
        #   different string: 'label', 'split','left','right' (See implementation below)
        #   'label': a label node with value as the mode of the labels
        #   'split': the best split node with value as a tuple of feature id and threshold
        #   'left','right': the left and right branch with value as the label node
        # X := Tensor of size (m,d) -- Batch of m examples of demension d
        # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
        # max_depth := int == the deepest level of the the decision tree
        # k := int -- the number of classes, with default value as 4
        # Return := tuple of (int, float) -- the feature id and threshold of the best split
        #

        H = self.entropy(y)
        if H == 0 or max_depth == 0:
            return

        best_split = self.find_split(node, X, y, k=k)

        ####################################
        # THIS LINE BELOW WILL REMOVE UNKNOWN OPCODE
        ####################################
        if best_split == None:
            return

        idx = X[:,best_split[0]] > best_split[1]
        X_left, y_left = X[idx], y[idx]
        X_right, y_right = X[~idx], y[~idx]

        del node['label']
        node['split'] = best_split
        node['left'] = { 'label': y_left.mode().values }
        node['right'] = { 'label': y_right.mode().values }

        # Fill in the following two lines to recursively build the rest of the
        # decision tree
        # self.expand_node(...)
        # self.expand_node(...)
        ## ANSWER
        self.expand_node(node['left'], X_left, y_left, max_depth-1, k)
        self.expand_node(node['right'], X_right, y_right, max_depth-1, k)
        ## END ANSWER

        return

    def predict_one(self, node, x):
        # Makes a prediction for a single example.
        # node := Map(string: value) -- the tree represented as a Map, the key will take four
        #   different string: 'label', 'split','left','right' (See implementation below)
        #   'label': a label node with value as the mode of the labels
        #   'split': the best split node with value as a tuple of feature id and threshold
        #   'left','right': the left and right branch with value as the label node
        # x := Tensor(float) of size(d,) -- the single example in a batch
        # Fill in the rest
        if 'label' in node:
            return node['label']
        else:
            feature, threshold = node['split']
            if x[feature] > threshold:
                return self.predict_one(node['left'], x)
            else:
                return self.predict_one(node['right'], x)

# %%
def fit_decision_tree(X,y, k=4):
    # The function will fit data with decision tree with the expand_node method implemented above

    root = { 'label': y.mode().values }
    dt = DecisionTree()
    dt.expand_node(root, X, y, max_depth=10, k=k)
    return root

def predict(node, X):
    # return the predict result of the entire batch of examples using the predict_one function above.
    dt = DecisionTree()
    return torch.stack([dt.predict_one(node, x) for x in X])

# %% [markdown]
# Test your code on the `iris` dataset. Your decision tree should fit to 100\% training accuracy and generalize to 88\% test accuracy.

# %%
iris = datasets.load_iris()
data=train_test_split(iris.data,iris.target,test_size=0.5,random_state=123)

X,X_te,y,y_te = [torch.from_numpy(A) for A in data]
X,X_te,y,y_te = X.float(), X_te.float(), y.long(), y_te.float()

DT = fit_decision_tree(X,y,k=4)
print('Train accuracy: ', (predict(DT, X) == y).float().mean().item())
print('Test accuracy: ', (predict(DT, X_te) == y_te).float().mean().item())

# %%

grader.grade(test_case_id = 'entropy', answer = get_class_source(DecisionTree))
grader.grade(test_case_id = 'find_split', answer = get_class_source(DecisionTree))
grader.grade(test_case_id = 'expand_node', answer = get_class_source(DecisionTree))
grader.grade(test_case_id = 'predict_one', answer = get_class_source(DecisionTree))

# %% [markdown]
# ### Bagging Decision Trees for Random forests
# 
# Note that our `find_split` implementation can use a random subset of the features when searching for the right split via the argument $k$. For the vanilla decision tree, we defaulted to $k=4$. Since there were 4 features, this meant that the decision tree could always considered all 4 features. By reducing the value of $k$ to $\sqrt(k)=2$, we can introduce variance into the decision trees for the bagging algorithm.
# 
# You'll now implement the bagging algorithm. Note that if you use the `clf` and `predict` functions given as keyword arguments, you can pass this section in the autograder without needing a correct implementation for decision trees from the previous section.
# 
# 1. Bootstrap (1pt): Implement `bootstrap` to draw a random bootstrap dataset from the given dataset.
# 2. Fitting a random forest (1pt): Implement `random_forest_fit` to train a random forest that fits the data.
# 3. Predicting with a random forest (1pt): Implement `predict_forest_fit` to make predictions given a random forest.
# 
# Tip:
# + If you're not sure whether your bootstrap is working or not, remember that on average, there will be $1-1/e\approx 0.632$ unique samples in a bootstrapped dataset.

# %%
def bootstrap(X,y):
    # Draw a random bootstrap dataset from the given dataset.
    #
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # y := Tensor(int) of size (m) -- the given vectors of labels of the examples
    #
    # Return := Tuple of (Tensor(float) of size (m,d),Tensor(int) of size(m,)) -- the random bootstrap
    #       dataset of X and its correcting lable Y
    # Fill in the rest
    m = X.size(0)
    selected_inex = torch.randint(0, m, (m,))
    return  X[selected_inex], y[selected_inex]

def random_forest_fit(X, y, m, k, clf, bootstrap_func):
    # Train a random forest that fits the data.
    # X := Tensor(float) of size (n,d) -- Batch of n examples of demension d
    # y := Tensor(int) of size (n) -- the given vectors of labels of the examples
    # m := int -- number of trees in the random forest
    # k := int -- number of classes of the features
    # clf := function -- the decision tree model that the data will be trained on
    # bootstrap := function -- the function to use for bootstrapping (pass in "bootstrap")
    #
    # Return := the random forest generated from the training datasets
    # Fill in the rest
    random_forest = []
    for i in range(m):
        X_b, y_b = bootstrap_func(X, y)
        tree = clf(X_b, y_b, k)
        random_forest.append(tree)
    return random_forest    

def random_forest_predict(X, clfs, predict):
    # Implement `predict_forest_fit` to make predictions given a random forest.
    # X := Tensor(float) of size (m,d) -- Batch of m examples of demension d
    # clfs := list of functions -- the random forest
    # predict := function that predicts (will default to your "predict" function)
    # Return := Tensor(int) of size (m,) -- the predicted label from the random forest
    # Fill in the rest
    prefictions = [predict(tree, X) for tree in clfs]
    prefictions = torch.stack(prefictions, dim=0)
    return prefictions.mode(dim=0).values

# %% [markdown]
# Test your code again on the `iris` dataset. Our random forest was able to improve the accuracy of the decision tree by about 10\%!

# %%

torch.manual_seed(42)
RF = random_forest_fit(X,y,50,2, clf=fit_decision_tree, bootstrap_func=bootstrap)

print('Train accuracy: ', (random_forest_predict(X, RF, predict) == y).float().mean().item())
print('Test accuracy: ', (random_forest_predict(X_te, RF, predict) == y_te).float().mean().item())

# %%

grader.grade(test_case_id = 'bootstrap', answer = getsource(bootstrap))
grader.grade(test_case_id = 'random_forest_fit', answer = getsource(random_forest_fit))
grader.grade(test_case_id = 'random_forest_predict', answer = getsource(random_forest_predict))

# %% [markdown]
# As a sanity check, the random forest can get around 95-97% test accuracy.

# %%



