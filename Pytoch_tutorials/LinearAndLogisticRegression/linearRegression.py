"""
To implement any machine learning algorithm we have to follow certain steps.
Step 1: Import libraries
Step 2: Prepare dataset
Step 3: Create model
Step 4: Initialize loss and optimizer
Step 5: Training
        : Forward pass - to calculate the loss
        : Backward pass- compute the gradient
        : Update - update the gradient
"""
# Step-1 Import libraries
import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#step -2 dataset preparation 
X_numpy, y_numpy = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

n_samples, n_features = X.shape
#print(n_samples, n_features)
y = y.view(y.shape[0],1)

#step- 3
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


#step -4 loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# step -5 training 
num_iters = 100

for epoch in range(num_iters):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    #backward pass
    loss.backward() #it calcuates gradients

    #update
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if (epoch+1)% 10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')

predicted = model(X).detach()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.savefig('linearRegression.png')
plt.show()
