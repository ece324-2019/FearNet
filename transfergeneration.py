import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data

import torch.optim as optim

import torchvision
from torchvision import models
import torchvision.transforms as transforms

#from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from PIL import Image
from dataclass import DataClass
from models import Baseline, DCNNEnsemble_3, resnet152, TransferEnsemble, vgg19bn, dense161, resnext101, wres101, alex, google, shuffle, TransferEnsembleFrozen
from metrics import accuracy, evaluate
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
torch.cuda.empty_cache()
# print(Ch1Mean, "\n", Ch1SD, "\n", Ch2Mean, "\n", Ch2SD, "\n", Ch3Mean, "\n", Ch3SD)
Ch1Mean = 0.4882
Ch1SD = 2850.2090/12735
Ch2Mean = 0.4723
Ch2SD = 2754.8560/12735
Ch3Mean = 0.4512
Ch3SD = 2778.6946/12735
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize((Ch1Mean,Ch2Mean,Ch3Mean),(Ch1SD,Ch2SD,Ch3SD))])

image_data = torchvision.datasets.ImageFolder(root='./data',transform=transform)

# imgloader = torch.utils.data.DataLoader(image_data, batch_size=4, shuffle=True, num_workers=4)

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

# print('label',labelohe)
# print('img',img_col)
X_trval, X_test, Y_trval, Y_test= train_test_split(img_col,labelohe, test_size=0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_trval,Y_trval,test_size=0.2)

train = DataClass(X_train,Y_train)
valid = DataClass(X_val,Y_val)
test = DataClass(X_test,Y_test)

bs = 64
e_num = 20
trainloader = DataLoader(train, shuffle=True, batch_size=bs,pin_memory=False)
valloader = DataLoader(valid,shuffle=True,batch_size=bs,pin_memory=False)
testloader = DataLoader(test,shuffle=True,batch_size=bs,pin_memory=False)

torch.manual_seed(1)
net0 = resnet152()
net1 = vgg19bn()
net2 = dense161()
net3 = alex()
net4 = resnext101()
net5 = google()
net6 = wres101()
net7 = shuffle()
net_arr = [net0,net1,net2,net3,net4,net5,net6,net7]

optimizer_arr = []
for i in range(0,len(net_arr)):
    optimizer_arr.append(optim.Adam(net_arr[i].parameters(),lr=0.001))
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

for tloopnum in range(len(net_arr)):
    net = net_arr[tloopnum]
    optimizer = optimizer_arr[tloopnum]

    val_acc_tot = []
    train_acc_tot = []
    loss_tot = []
    val_loss_tot = []
    f1_tot = []
    recall_tot = []
    n_tot = []
    j = 0
    for epoch in range(e_num):
        net = net.train()
        sum_loss = 0
        running_loss = 0
        running_acc = 0
        batchperepoch = 0
        for i, data in enumerate(trainloader,0):
            batchperepoch += 1
            j += 1
            print('Model #: ', tloopnum)
            print('Epoch #: ', epoch)
            print('Batch #: ', batchperepoch)
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            # print('batch label',labels)
            # print('batch_label long',labels.long())
            loss =  criterion(outputs.squeeze(),torch.max(labels.long(),1)[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.detach()
            running_acc += accuracy(outputs, labels)[0]

        # if j%step== 0:
        # if epoch%1 == epoch:
        net = net.eval()
        loss_tot.append(running_loss/batchperepoch)
        # t_acc = accuracy(outputs,labels)[0]
        t_acc = running_acc/batchperepoch
        train_acc_tot.append(t_acc)
        # acc_tot.append(acc[0])
        n_tot.append(epoch)
        running_loss = 0
        temp = evaluate(net, valloader)
        val_loss_tot.append(temp[1])
        val_acc_tot.append(temp[0])
        print('Model #: ', tloopnum)
        print('Train Acc: ', t_acc)
        print('Validation Acc: ', temp[0])
        print('Train Loss: ', running_loss/batchperepoch)
        print('Validation Loss: ', temp[1])
        y_ground = []
        y_pred = []
        for j, batch in enumerate(valloader, 1):
            valid_train, valid_label = batch
            predict = net(valid_train.float())
            predictions = predict.detach()
            index = 0
            for pred in predictions:
                p_val, p_clas = torch.max(pred, 0)
                v_val, v_clas = torch.max(valid_label[index], 0)
                y_pred.append(p_clas.item())
                y_ground.append(v_clas.item())
                index += 1
        f1 = f1_score(y_ground, y_pred, average='micro')
        recall = recall_score(y_ground, y_pred, average='micro')
        print('F1 :', f1)
        print('Recall: ', recall)
        f1_tot.append(f1)
        recall_tot.append(recall)
    print('Finished Training')
    print('Train Acc: ', train_acc_tot)
    print('Val Acc: ', val_acc_tot)
    print('Train Loss: ', loss_tot)
    print('Valid Loss: ', val_loss_tot)
    print('Recall Score: ', recall_tot)
    print('F1 Score: ', f1_tot)
    # y_ground = []
    # y_pred = []
    # for j, batch in enumerate(testloader, 1):
    #     valid_train, valid_label = batch
    #     predict = net(valid_train.float())
    #     predictions = predict.detach()
    #     index = 0
    #     for pred in predictions:
    #         p_val, p_clas = torch.max(pred, 0)
    #         v_val, v_clas = torch.max(valid_label[index], 0)
    #         y_pred.append(p_clas.item())
    #         y_ground.append(v_clas.item())
    #         index += 1
    # # print('y_ground', y_ground)
    # # print('y_pred', y_pred)
    # print(confusion_matrix(y_ground,y_pred))
    # torch.save(net, 'model%d.pt' % tloopnum)
    if tloopnum == 0:
        torch.save(net, 'resnet.pt')
    elif tloopnum == 1:
        torch.save(net, 'vgg.pt')
    elif tloopnum == 2:
        torch.save(net, 'dense.pt')
    elif tloopnum == 3:
        torch.save(net, 'alex.pt')
    elif tloopnum == 4:
        torch.save(net, 'resnext.pt')
    elif tloopnum == 5:
        torch.save(net, 'google.pt')
    elif tloopnum == 6:
        torch.save(net, 'wres.pt')
    elif tloopnum == 7:
        torch.save(net, 'shuffle.pt')
    else:
        print('ERROR: Non-existent model!')

# Now, all transfer models have been finetuned for our problem.
# Using these, we will attempt to assemble
net0 = torch.load('resnet.pt')
net1 = torch.load('vgg.pt')
net2 = torch.load('dense.pt')
net3 = torch.load('alex.pt')
net4 = torch.load('resnext.pt')
net5 = torch.load('google.pt')
net6 = torch.load('wres.pt')
net7 = torch.load('shuffle.pt')

net = TransferEnsembleFrozen(net0,net1,net2,net3,net4,net5,net6,net7)
optimizer = optim.Adam(net.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

val_acc_tot = []
train_acc_tot = []
test_acc_tot = []
loss_tot = []
val_loss_tot = []
test_loss_tot = []
n_tot = []
f1_tot = []
recall_tot = []
j = 0
for epoch in range(e_num):
    net = net.train()
    sum_loss = 0
    running_loss = 0
    running_acc = 0
    batchperepoch = 0
    for i, data in enumerate(trainloader, 0):
        batchperepoch += 1
        j += 1
        print('Epoch #: ', epoch)
        print('Batch #: ', batchperepoch)
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        # print('batch label',labels)
        # print('batch_label long',labels.long())
        loss = criterion(outputs.squeeze(), torch.max(labels.long(), 1)[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.detach()
        running_acc += accuracy(outputs, labels)[0]

    # if j%step== 0:
    # if epoch%1 == epoch:
    net = net.eval()
    print(running_loss / batchperepoch)
    loss_tot.append(running_loss / batchperepoch)
    # t_acc = accuracy(outputs,labels)[0]
    t_acc = running_acc / batchperepoch
    train_acc_tot.append(t_acc)
    # acc_tot.append(acc[0])
    n_tot.append(epoch)
    running_loss = 0
    temp = evaluate(net, valloader)
    testeval = evaluate(net, testloader)
    val_loss_tot.append(temp[1])
    val_acc_tot.append(temp[0])
    test_acc_tot.append(testeval[0])
    test_loss_tot.append(testeval[1])
    print('Train Acc: ', t_acc)
    print('Validation Acc: ', temp[0])
    print('Testing Acc: ',testeval[0])
    print('Train Loss: ', running_loss / batchperepoch)
    print('Validation Loss: ', temp[1])
    print('Testing Loss: ', testeval[1])
    y_ground = []
    y_pred = []
    for j, batch in enumerate(valloader, 1):
        valid_train, valid_label = batch
        predict = net(valid_train.float())
        predictions = predict.detach()
        index = 0
        for pred in predictions:
            p_val, p_clas = torch.max(pred, 0)
            v_val, v_clas = torch.max(valid_label[index], 0)
            y_pred.append(p_clas.item())
            y_ground.append(v_clas.item())
            index += 1
    f1 = f1_score(y_ground, y_pred, average='micro')
    recall = recall_score(y_ground, y_pred, average='micro')
    print('F1 :', f1)
    print('Recall: ', recall)
    f1_tot.append(f1)
    recall_tot.append(recall)
print('Finished Training')
print('Train Acc: ', train_acc_tot)
print('Val Acc: ', val_acc_tot)
print('Test Acc: ', test_acc_tot)
print('Train Loss: ', loss_tot)
print('Valid Loss: ', val_loss_tot)
print('Test Loss: ', test_loss_tot)
print('Recall Score: ', recall_tot)
print('F1 Score: ', f1_tot)
y_ground = []
y_pred = []

for j, batch in enumerate(testloader, 1):
    valid_train, valid_label = batch
    predict = net(valid_train.float())
    predictions = predict.detach()
    index = 0
    for pred in predictions:
        p_val, p_clas = torch.max(pred, 0)
        v_val, v_clas = torch.max(valid_label[index], 0)
        y_pred.append(p_clas.item())
        y_ground.append(v_clas.item())
        index += 1
print('y_ground', y_ground)
print('y_pred: ', y_pred)
print(confusion_matrix(y_ground, y_pred))
print('F1 :', f1_score(y_ground, y_pred))
print('Recall: ', recall_score(y_ground, y_pred))