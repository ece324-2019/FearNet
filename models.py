import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, size):
        super(Baseline, self).__init__()
        self.size = size
        # First convolutional layer: 50 3x3 kernels
        # 64x64x3 -> 62x62x50
        self.conv1 = nn.Conv2d(3,50,3)
        # First max pooling layer: 2x2 w/ stride = 2
        # 31x31x50
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 50 5x5 kernels
        # 29x29x50
        self.conv2 = nn.Conv2d(50,50,5)
        # Second max pooling layer: 2x2 w/ stride = 2 w/ padding
        # 13x13x50
        # Batch Normalization on convolutional layers
        self.conv_BN = nn.BatchNorm2d(50)
        # Fully connected output layer
        # First layer = 1x8450 input
        self.fc1 = nn.Linear(8450, 3)
        # Fully connected batch normalization
        self.fc1_BN = nn.BatchNorm1d(3)


    def forward(self,x):

        # Convolution, Batch Normalization, and Pooling
        x = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv_BN(self.conv2(x))))

        # print('post',x.size())
        x = x.view(-1,8450)
        # print(x.size())
        # Fully connected output
        x = F.sigmoid(self.fc1_BN(self.fc1(x)))
        return x

class DCNN(nn.Module):
    def __init__(self, size):
        super(DCNN, self).__init__()

    def forward(self,x):
        return x
