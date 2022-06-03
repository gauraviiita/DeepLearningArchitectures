import torch
#-----------------------------

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_() # this function stop the adding the value with previous one, for seeing
    # deference between both just comment this line
    
"""
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()

optimizer.zero_grad()
"""
# example two