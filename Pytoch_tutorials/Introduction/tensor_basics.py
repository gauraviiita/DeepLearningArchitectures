from numpy import dtype, require
import torch

#Creating empty tensor
x = torch.empty(3)
print(x)

#crating a two diemnsional tensor
twoD = torch.empty(2,5)
print(twoD)

#creating a three dimension tensor
threeD = torch.empty(3,3,4)
print(threeD)

#creating a four dimensional tensor
fourD = torch.empty(2, 2, 2, 3)
print(fourD)

#creating a five dimensional tensor
fiveD = torch.empty(3, 4, 2, 2, 5)
print(fiveD)

#creating a random tensor of two dimensions
randomval = torch.rand(2,2)
print(randomval)
# creating a zero tensor
zerosnum = torch.zeros(2,2)
print(zerosnum)

#creating one tensor
onesnum = torch.ones(2,2)
print(onesnum.dtype)

onenew = torch.ones(2,3, dtype=torch.float16)
print(onenew.size())


tens = torch.tensor([2.3, 5,3, 8.0, 2])
print(tens)

# tensor operations
# addition of two tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y
print(z)
z1 = torch.add(x, y)
print(z1)

y.add_(x)
print(y)


# subtraction of two tensor
c = x - y
print(c)
c1 = torch.sub(x, y)
print(c1)

#multiplication operation 
m = x*y
print(m)

#division operation
d = x/y
print(d)



# slicing operations
x = torch.rand(5, 3)
print(x)

print(x[1:,0])
print(x[2,1])
print(x[2,2].item()) # it gives actual value and only when tensor has single element in it.

#reshaping a tensor

x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
y1 = x.view(2,8)
print(y1)
y2 = x.view(-1, 4)
print(y2)
y3 = x.view(-1, 8) # it automatically detect the number of row, and we have to give only #colums
print(y3)

print(y3.size())

#converty from numpy to torch and viceversa

import numpy as np

# first tensor to numpy 
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
print(a)
print(b)

#now array to tensor

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b) # inplace operation allocate same address in cpu. so if we increase a by one then b will also increase


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y= y.to(device)
    z = x + y
    z = z.to("cpu")


x = torch.ones(5, requires_grad=True)
print(x)