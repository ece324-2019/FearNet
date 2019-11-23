import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        self.fc1 = nn.Linear(42050, 18)
        # Fully connected batch normalization
        self.fc1_BN = nn.BatchNorm1d(18)


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

class DCNNEnsemble_3(nn.Module):
    def __init__(self):
        super(DCNNEnsemble_3, self).__init__()
        # self.size = size
        self.conv1 = nn.Conv2d(3, 50, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 50, 5)
        self.conv_BN = nn.BatchNorm2d(50)

        self.fc1 = nn.Linear(42050, 512)
        self.fc2 = nn.Linear(512,18)
        # self.fc3 = nn.Linear(64,18)

        self.fc1_BN = nn.BatchNorm1d(512)
        self.fc2_BN = nn.BatchNorm1d(18)
        # self.fc3_BN = nn.BatchNorm1d(18)


    def forward(self,x):
        z = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        # z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        # z = self.pool(F.relu(self.conv_BN(self.conv2(z))))
        z = z.view(-1, 42050)
        z = F.relu(self.fc1_BN(self.fc1(z)))
        z = self.fc2_BN(self.fc2(z))

        y = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        # y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        # y = self.pool(F.relu(self.conv_BN(self.conv2(y))))
        y = y.view(-1, 42050)
        y = F.relu(self.fc1_BN(self.fc1(y)))
        y = self.fc2_BN(self.fc2(y))

        w = self.pool(F.relu(self.conv_BN(self.conv1(x))))
        w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        # w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        # w = self.pool(F.relu(self.conv_BN(self.conv2(w))))
        w = w.view(-1, 42050)
        w = F.relu(self.fc1_BN(self.fc1(w)))
        w = self.fc2_BN(self.fc2(w))

        x = (z+y+w)/3
        return x

class resnet152(nn.Module):
    def __init__(self):
        super(resnet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        print(num_ftrs)
        self.model.fc = nn.Linear(num_ftrs,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,18)

    def forward(self,x):
        x = F.relu(self.model(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class vgg19bn(nn.Module):
    def __init__(self):
        super(vgg19bn, self).__init__()
        self.model = models.vgg19_bn(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier[6].in_features
        old = list(self.model.classifier.children())
        old.pop()
        old.append(nn.Linear(num_ftrs,18))
        self.model.classifier = nn.Sequential(*old)
        # self.fc2 = nn.Linear(2048,18)
        # self.fc3 = nn.Linear(1024,18)

    def forward(self,x):
        x = self.model(x)
        # x = F.relu(self.model(x))
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = models.alexnet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier[6].in_features
        old = list(self.model.classifier.children())
        old.pop()
        old.append(nn.Linear(num_ftrs,18))
        print(num_ftrs)
        self.model.classifier = nn.Sequential(*old)
        # self.fc2 = nn.Linear(2048,18)
        # self.fc3 = nn.Linear(1024,18)

    def forward(self,x):
        x = self.model(x)
        # x = F.relu(self.model(x))
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

class dense161(nn.Module):
    def __init__(self):
        super(dense161, self).__init__()
        self.model = models.densenet161(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs,1104)
        self.fc2 = nn.Linear(1104, 18)
        self.bn2 = nn.BatchNorm1d(18)
        self.bn = nn.BatchNorm1d(1104)
        #self.fc3 = nn.Linear(552,18)

    def forward(self, x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
        # x = F.relu(self.model(x))
        #x = F.relu(self.fc2(x))
        # x = self.fc2(x)
        return x

class resnext101(nn.Module):
    def __init__(self):
        super(resnext101, self).__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        print(num_ftrs)
        self.model.fc = nn.Linear(num_ftrs,1024)
        self.fc2 = nn.Linear(1024, 18)
        # self.fc3 = nn.Linear(512, 18)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(18)


    def forward(self,x):
        x = F.relu(self.model(x))
        # x = self.bn2(self.fc2(x))
        x = self.fc2(x)

        return x

class wres101(nn.Module):
    def __init__(self):
        super(wres101, self).__init__()
        self.model = models.wide_resnet101_2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        print(num_ftrs)
        self.model.fc = nn.Linear(num_ftrs,1024)
        self.fc2 = nn.Linear(1024,18)
        self.bn = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(18)
    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))

        return x

class avgTransferEnsemble(nn.Module):
    def __init__(self):
        super(avgTransferEnsemble, self).__init__()
        self.res152 = models.resnet152(pretrained=True)
        self.vgg19bn = models.vgg19_bn(pretrained=True)
        self.dense161 = models.densenet161(pretrained=True)
        self.wres101 = models.wide_resnet101_2(pretrained=True)
        self.resnext101 = models.resnext101_32x8d(pretrained=True)

        # Freezing

        # Finding size of last layer
        n1 = self.res152.fc.in_features
        n2 = self.vgg19bn.classifier[6].in_features
        n3 = self.dense161.fc.in_features
        n5 = self.wres101.fc.in_features
        n6 = self.resnext101.fc.in_features

        # Some syntax to prep vgg19_bn for last fc layer replacement
        # Convert all vgg19_bn layers to list and remove last one
        features = list(self.vgg19bn.classifier.children())[:-1]
        # Add the last layer based on the num of classes in our dataset
        features.extend([nn.Linear(n2, 18)])
        # Replacing last fc layer for all transfer models
        self.res152.fc = nn.Linear(n1, 18)
        self.vgg19bn = nn.Sequential(*features)
        self.dense161.fc = nn.Linear(n3, 18)
        self.inceptv3.fc = nn.Linear(n4, 18)
        self.wres101.fc = nn.Linear(n5, 18)
        self.resnext50.fc = nn.Linear(n6, 18)

    def forward(self,x):
        x1 = self.res152(x)
        x2 = self.vgg19bn(x)
        x3 = self.dense161(x)
        x4 = self.inceptv3(x)
        x5 = self.wres101(x)
        x6 = self.resnext50(x)

        z = (x1+x2+x3+x4+x5+x6)/6
        return z
