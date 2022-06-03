# what is gradient 
# it is essential for model optimization
from tkinter.tix import Tree

from numpy import dtype, require
import torch




x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2

print(y)

z = y*y*2

print(z)
#z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad)


#-----------------------------------#

#x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

a = torch.randn(3, requires_grad=True)
print(a)

a1 = a.requires_grad_(False)
print(a1)


b = a.detach()
print(b)

with torch.no_grad():
    c = a + 2
    print(c)


#-----------------------------

weights = torch.ones(4, requires_grad=True)

for epoch in range(1):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)