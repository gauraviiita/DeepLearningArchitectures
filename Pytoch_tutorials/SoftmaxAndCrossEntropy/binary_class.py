from turtle import forward
from multi_class import NeuralNet2
import torch 
import torch.nn as nn

# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidde_size1, hidde_size2):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidde_size1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidde_size1, hidde_size2)
        self.linear3 = nn.Linear(hidde_size2, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        #sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidde_size1=10, hidde_size2=5)
criterion = nn.BCELoss()
