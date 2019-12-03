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

        # print('post',x.size())
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
        self.fc2 = nn.Linear(1024,17)
        self.bn = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(17)
        # self.fc3 = nn.Linear(512,18)

    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
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
        old.append(nn.Linear(num_ftrs,2048))
        self.model.classifier = nn.Sequential(*old)
        self.fc2 = nn.Linear(2048,17)
        self.bn = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(17)
        # self.fc3 = nn.Linear(1024,18)

    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
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
        old.append(nn.Linear(num_ftrs,2048))
        print(num_ftrs)
        self.model.classifier = nn.Sequential(*old)
        self.fc2 = nn.Linear(2048,17)
        self.bn = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(17)

    def forward(self,x):
        # x = self.model(x)
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
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
        self.fc2 = nn.Linear(1104, 17)
        self.bn2 = nn.BatchNorm1d(17)
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
        self.model.fc = nn.Linear(num_ftrs,1024)
        self.fc2 = nn.Linear(1024, 17)
        self.bn = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(17)

    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
        return x

class google(nn.Module):
    def __init__(self):
        super(google, self).__init__()
        self.model = models.googlenet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 17)
        self.bn = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(17)

    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
        # x = self.bn2(self.fc2(x))
        # x = self.fc2(x)

        return x

class wres101(nn.Module):
    def __init__(self):
        super(wres101, self).__init__()
        self.model = models.wide_resnet101_2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,1024)
        self.fc2 = nn.Linear(1024,17)
        self.bn = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(17)
    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))

        return x

class shuffle(nn.Module):
    def __init__(self):
        super(shuffle, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 17)
        self.bn = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(17)

    def forward(self,x):
        x = F.relu(self.bn(self.model(x)))
        x = self.bn2(self.fc2(x))
        # x = self.model(x)
        return x

# class squeeze(nn.Module):
#     def __init__(self):
#         super(squeeze, self).__init__()
#         self.model = models.squeezenet1_1(pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad = False
#         # num_ftrs = self.model.fc.in_features
#         old = list(self.model.classifier.children())
#         old.pop()
#         old.append(nn.Linear(1000,18))
#         # print(num_ftrs)
#         self.model.classifier = nn.Sequential(*old)
#         # self.fc2 = nn.Linear(2048,18)
#         # self.bn = nn.BatchNorm1d(2048)
#         # self.bn2 = nn.BatchNorm1d(18)
#
#     def forward(self,x):
#         x = self.model(x)
#         # x = F.relu(self.bn(self.model(x)))
#         # x = self.bn2(self.fc2(x))
#         # x = self.fc3(x)
#         return x

class TransferEnsemble(nn.Module):
    def __init__(self):
        super(TransferEnsemble, self).__init__()
        self.res152 = resnet152()
        self.vgg19bn = vgg19bn()
        self.dense161 = dense161()
        self.alex = alex()
        self.resnext101 = resnext101()
        self.google = google()
        self.wres101 = wres101()
        self.shuffle = shuffle()

        self.fc1 = nn.Linear(144,72)
        self.fc2 = nn.Linear(68,18)
        self.bn1 = nn.BatchNorm1d(68)
        self.bn2 = nn.BatchNorm1d(18)


    def forward(self,x):
        # x1 = self.res152(x)
        # x2 = self.vgg19bn(x)
        # x3 = self.dense161(x)
        # x4 = self.alex(x)
        # x5 = self.resnext101(x)
        # x6 = self.google(x)
        # x7 = self.wres101(x)
        # x8 = self.shuffle(x)

        x1 = F.relu(self.res152(x))
        x2 = F.relu(self.vgg19bn(x))
        x3 = F.relu(self.dense161(x))
        x4 = F.relu(self.alex(x))
        x5 = F.relu(self.resnext101(x))
        x6 = F.relu(self.google(x))
        x7 = F.relu(self.wres101(x))
        x8 = F.relu(self.shuffle(x))

        z = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.bn2(self.fc2(z))
        # z = (x1+x2+x3+x4+x5+x6+x7+x8)/8
        return z

class TransferEnsembleFrozen(nn.Module):
    def __init__(self, m1, m2, m3, m4, m5, m6, m7, m8):
        super(TransferEnsembleFrozen,self).__init__()
        self.marray = [m1,m2,m3,m4,m5,m6,m7,m8]
        for model in self.marray:
            for param in model.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(136,68)
        self.fc2 = nn.Linear(68,18)
        self.bn2 = nn.BatchNorm1d(18)
        self.bn1 = nn.BatchNorm1d(68)

    def forward(self,x):
        x1 = self.marray[0](x)
        x2 = self.marray[1](x)
        x3 = self.marray[2](x)
        x4 = self.marray[3](x)
        x5 = self.marray[4](x)
        x6 = self.marray[5](x)
        x7 = self.marray[6](x)
        x8 = self.marray[7](x)
        z = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.bn2(self.fc2(z))
        return z

class TransferEnsembleFrozenLight(nn.Module):
    def __init__(self, m1, m2, m3):
        super(TransferEnsembleFrozenLight,self).__init__()
        self.marray = [m1,m2,m3]
        for model in self.marray:
            for param in model.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(51,68)
        self.fc2 = nn.Linear(68,18)
        self.bn2 = nn.BatchNorm1d(18)
        self.bn1 = nn.BatchNorm1d(68)

    def forward(self,x):
        x1 = self.marray[0](x)
        x2 = self.marray[1](x)
        x3 = self.marray[2](x)

        z = torch.cat((x1,x2,x3),1)
        z = F.relu(self.bn1(self.fc1(z)))
        z = self.bn2(self.fc2(z))
        return z