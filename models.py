import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, size):
        super(Baseline, self).__init__()
        self.size = size
        # First convolutional layer: 50 3x3 kernels
        self.conv1 = nn.Conv2d(3,50,3)
        # Second convolutional layer: 50 5x5 kernels
        self.conv2 = nn.Conv2d(50,50,5)
        # 2x2 Max Pooling w/ Stride = 2
        self.pool = nn.MaxPool2d(2,2)
        # Batch Normalization on convolutional layers
        self.conv_BN = nn.BatchNorm2d(50)
        # Fully connected output layer
        self.fc1 = nn.Linear(128, 1)
        # Fully connected batch normalization
        self.fc1_BN = nn.BatchNorm1d(128)


    def forward(self,x):
        # Convolution, Batch Normalization, and Pooling
        x = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv_BN(self.conv2(x))))
        x = x.view(-1,self.size**2)
        # Fully connected output
        x = F.sigmoid(self.fc1_BN(self.fc1(x)))
        return x
