""" 
    step1: forward pass: compute loss
    Step2: compute local gradients
    step3: backward pass: compute dLoss/dWeights using the chain rule 


"""

import torch 
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass to compute loss
y_hat = w * x
loss = (y_hat -y)**2
print(loss)

#backward pass

loss.backward()
print(w.grad)


## update our weight
#next forward and backward pass

for epoch in range(3):
    
    #forward pass
    y_hat = w*x
    loss = (y_hat - y)**2
    print(loss)
    loss.backward()
    print(w.grad)
    #w.grad.zero_()
