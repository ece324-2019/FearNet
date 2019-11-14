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
from PIL import Image
from dataclass import DataClass
from models import Baseline

# --- CALCULATING IMAGE CHANNEL MEANS AND STANDARD DEVIATIONS---
# transform = transforms.Compose([transforms.ToTensor()])
# AllImages = torchvision.datasets.ImageFolder(root='.',transform=transform)
# imgloader = DataLoader(AllImages, shuffle =True, batch_size=1)
#
# Ch1Mean = 0.0
# Ch1SD = 0.0
# Ch2Mean = 0.0
# Ch2SD = 0.0
# Ch3Mean = 0.0
# Ch3SD = 0.0
#
# counter = 1
# for img in imgloader:
#     print("Image Counter: ",counter)
#     Ch1Mean += img[0][0][0].mean()
#     Ch1SD += img[0][0][0].std()
#     Ch2Mean += img[0][0][1].mean()
#     Ch2SD += img[0][0][1].std()
#     Ch3Mean += img[0][0][2].mean()
#     Ch3SD += img[0][0][2].std()
#     print('Ch1', Ch1SD, 'Ch2', Ch2SD, 'Ch3', Ch3SD)
#     counter += 1
#
# Ch1Mean = Ch1Mean/len(AllImages)
# Ch1SD = Ch1SD/len(AllImages)
# Ch2Mean = Ch2Mean/len(AllImages)
# Ch2SD = Ch2SD/len(AllImages)
# Ch3Mean = Ch3Mean/len(AllImages)
# Ch3SD = Ch3SD/len(AllImages)
# print('Ch1',Ch1SD,'Ch2',Ch2SD,'Ch3',Ch3SD)
#
# print(Ch1Mean, "\n", Ch1SD, "\n", Ch2Mean, "\n", Ch2SD, "\n", Ch3Mean, "\n", Ch3SD)
Ch1Mean = 0.4882
Ch1SD = 2850.2090/12735
Ch2Mean = 0.4723
Ch2SD = 2754.8560/12735
Ch3Mean = 0.4512
Ch3SD = 2778.6946/12735

transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((Ch1Mean,Ch2Mean,Ch3Mean),(Ch1SD,Ch2SD,Ch3SD))])

image_data = torchvision.datasets.ImageFolder(root='./data',transform=transform)

imgloader = torch.utils.data.DataLoader(image_data, batch_size=4, shuffle=True, num_workers=4)

# dataiter = iter(AllImagesLoader)
# images, labels = dataiter.next()

# ---ONEHOTENCODING IMAGE LABELS---
# Images are already classified with a number between 0-20, by the folder within which they were found in.

img_col = []
label_col = []
for i in image_data:
    img_col.append(i[0].numpy())
    label_col.append(i[1])

df_label = pd.DataFrame(label_col,columns=["phobia_type"])
ohe = OneHotEncoder(categories="auto")
M = df_label["phobia_type"].to_numpy().reshape(-1,1)
X = ohe.fit_transform(M).toarray()
dfOneHot = pd.DataFrame(X)
labelohe = dfOneHot.to_numpy()

print('label',labelohe)
print('img',img_col)
X_train, X_test, Y_train, Y_test = train_test_split(img_col,labelohe, test_size=0.2)

train = DataClass(X_train,Y_train)
valid = DataClass(X_test,Y_test)

bs = 32
e_num = 1
trainloader = DataLoader(train, shuffle=True, batch_size=bs,pin_memory=False)
valloader = DataLoader(valid,shuffle=True,batch_size=len(Y_test),pin_memory=False)

#
# torch.manual_seed(1)
net = Baseline(64)
# summary(net,(3,56,56))
criterion = nn.BCEWithLogitsLoss
optimizer = optim.Adam(net.parameters(), lr=0.001)
#
#
# val_acc_tot = []
# train_acc_tot = []
# loss_tot = []
# val_loss_tot = []
# n_tot = []
# j = 0
# start = time.time()
for epoch in range(e_num):
    running_loss = 0
    for i, data in enumerate(trainloader,0):
        j += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        print('batch label',labels)
        # loss =  criterion(outputs.squeeze(),???)
        # loss.backward()
        # optimizer.step()
#
#         if j%per_epoch == 0:
            # print and append loss/acc stuff

# plot loss/acc stuff