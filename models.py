import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        # self.size = size
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
        self.fc1 = nn.Linear(42050, 19)
        # Fully connected batch normalization
        self.fc1_BN = nn.BatchNorm1d(19)


    def forward(self,x):

        # Convolution, Batch Normalization, and Pooling
        x = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        x = self.pool(F.relu(self.conv_BN(self.conv2(x))))

        print('post',x.size())
        x = x.view(-1,42050)
        # print(x.size())
        # Fully connected output
        x = F.sigmoid(self.fc1_BN(self.fc1(x)))
        return x

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        # self.size = size
        self.conv1 = nn.Conv2d(3, 50, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 50, 5)
        self.conv_BN = nn.BatchNorm2d(50)

        self.fc1 = nn.Linear(42050, 512)
        self.fc2 = nn.Linear(512,19)
        # self.fc3 = nn.Linear(64,19)

        self.fc1_BN = nn.BatchNorm1d(512)
        self.fc2_BN = nn.BatchNorm1d(19)
        # self.fc3_BN = nn.BatchNorm1d(19)


    def forward(self,x):
        z = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        z = z.view(-1, 42050)
        z = F.relu(self.fc1_BN(self.fc1(z)))
        z = F.sigmoid(self.fc2_BN(self.fc2(z)))

        y = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        y = y.view(-1, 42050)
        y = F.sigmoid(self.fc1_BN(self.fc1(y)))
        y = F.sigmoid(self.fc2_BN(self.fc2(y)))

        w = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        w = w.view(-1, 42050)
        w = F.sigmoid(self.fc1_BN(self.fc1(w)))
        w = F.sigmoid(self.fc2_BN(self.fc2(w)))

        x = (z+y+w)/3
        return x
