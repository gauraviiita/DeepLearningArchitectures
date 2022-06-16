#step -1 import libraries
from filecmp import cmp
from random import shuffle
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784 #28x28
hidde_size1 = 150
hidde_size2 = 50
num_class =10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

#step -3 load dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                transform = transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                shuffle=False)

# look one batch of data
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#plot the data
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
    plt.savefig('img.png')
plt.show()