import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model import CNN
import time


# Finding mean and standard deviation of each R,G, and B channel of the image datasets

# bs = 1
# transform = transforms.Compose([transforms.ToTensor()])
# image_data = torchvision.datasets.ImageFolder('phobiaimages',transform=transform)
# dataloader = DataLoader(image_data, shuffle =True, batch_size=bs)
#
# idx = 0
# ch_mean_accum = 0
# ch_std_accum = 0
# for i in dataloader:
#     idx += 1
#     print(idx)
#     ch_mean_accum += i[0][0][2].mean()
#     ch_std_accum += i[0][0][2].std()
#     print(i[0][0][0].mean())
#     print(i[0][0][0].std())
# print('R_mean',ch_mean_accum/idx)
# print('R_std',ch_std_accum/idx)
#
# R_mean =
# G_mean =
# B_mean =
# R_std =
# G_std =
# B_std =
#
# # Load Data
# transform = transforms.Compose(
#     transforms.ToTensor(),transforms.Normalize((R_mean,G_mean,B_mean),(R_std,G_std,B_std))]
# )
#
# # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.6233,0.5623,0.4944),(0.1214,0.1603,0.1643))])
# image_data = torchvision.datasets.ImageFolder('asl_images/asl_images',transform=transform)


img_col = []
label_col = []
for i in image_data:
    img_col.append(i[0].numpy())
    label_col.append(i[1])

df_label = pd.DataFrame(label_col,columns=["letter"])
ohe = OneHotEncoder(categories="auto")
M = df_label["letter"].to_numpy().reshape(-1,1)
X = ohe.fit_transform(M).toarray()
dfOneHot = pd.DataFrame(X)

labelohe = dfOneHot.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(img_col,labelohe, test_size=0.2)

train = DataClass(X_train,Y_train)
valid = DataClass(X_test,Y_test)

bs = 32
trainloader = DataLoader(train, shuffle=True, batch_size=bs,pin_memory=False)
valloader = DataLoader(valid,shuffle=True,batch_size=len(Y_test),pin_memory=False)

classes = ['A','B','C','D','E','F','G','H','I','K']


torch.manual_seed(5)
net = CNN()
summary(net,(3,56,56))
criterion = nn.BCEWithLogitsLoss
optimizer = optim.SGD(net.parameters(), lr=0.1)


val_acc_tot = []
train_acc_tot = []
loss_tot = []
val_loss_tot = []
n_tot = []
j = 0
start = time.time()
for epoch in range(e_num):
    running_loss = 0
    for i, data in enumerate(trainloader,0)
        j += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss =  criterion(outputs.squeeze(),???)
        loss.backward()
        optmizer.step()

        if j%per_epoch == 0:
            # print and append loss/acc stuff

# plot loss/acc stuff