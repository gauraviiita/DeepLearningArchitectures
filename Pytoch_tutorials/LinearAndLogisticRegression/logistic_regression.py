"""
    There are five steps to implement a machine learning algorithms
    Step-1 Import libraries
    Step-2 prepare dataset
    Step-3 initialization of model
    Step-4 Loss and optimizer
    Step-5 Training
            Forward pass
            Backward pass
            update
"""

#Step-1 import libraries
from distutils.log import Log
from tkinter.tix import X_REGION
from turtle import forward
import torch 
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#Step-2 Dataset preparation
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
#print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#scale the features
sc = StandardScaler() # when we implement logistic regression then always recommendation is to do zero mean and 1 std

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshape y
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# step -3 model

# f= wx + b and sigmoid

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# step -4 loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    #backward pass
    loss.backward()

    #update
    optimizer.step()

    #empty our gradient
    optimizer.zero_grad()

    if (epoch +1 )%10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')
    
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')