# design model (input, output, size, forward pass)
#2 construct the loss and optimizer
#3 training loop
#       forward pass: compute prediction
    #   backward pass: gradients
    #   update

from tkinter import W
from numpy import dtype
from gradeint_tutorial_torch import forward
import torch 
import torch.nn as nn


# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
#print(n_samples, n_features)
#exit()

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size, output_size)
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)



print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.001
n_iters = 1000

#loss function
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):
    y_pred = model(X)

    #loss 
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw
    #update weights
    optimizer.step()
    #zero_gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epcoh {epoch + 1}: w = {w[0][0].item():.3f}, loss={l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')