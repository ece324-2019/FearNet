import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
from sklearn.metrics import confusion_matrix
from PIL import Image
from dataclass import DataClass
from models import Baseline, DCNNEnsemble_3, resnet152, TransferEnsemble, vgg19bn, dense161, resnext101, wres101, alex, google, shuffle
from metrics import accuracy, evaluate

from guitest import image_loader
import os

AllPhobias = ['Heights','Open Spaces','Spiders','Lightning','Loneliness','Confined Spaces','Clowns','Dogs','Skin Defects','Vomit','Blood','Water','Birds','Snakes','Death','Needles','Holes']
ApplicablePhobias = []
# All the stuff inside your window.
layout = [  [sg.Text('Please select which of the following phobias apply to you:')],
            [sg.Button('Heights'), sg.Button('Open Spaces'),sg.Button('Spiders'),sg.Button('Lightning'),sg.Button('Confined Spaces'),sg.Button('Clowns'),sg.Button('Dogs'),sg.Button('Skin defects'),sg.Button('Vomit'),sg.Button('Blood'),sg.Button('Water'),sg.Button('Birds'),sg.Button('Snakes'),sg.Button('Death'),sg.Button('Needles'),sg.Button('Holes'),sg.Button('None')] ]

# Create the Window
window = sg.Window('Select', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.Read()
    if event in (None, 'None'):	# if user closes window or clicks cancel
        break
    if event not in ApplicablePhobias:
        ApplicablePhobias += [event]

window.close()
print(ApplicablePhobias)

def AskUser(im,phobia):
    layout = [[sg.Text("The following potentially disturbing content has been detected in this image:")],[sg.Text(phobia)], [sg.Text("Would you like to view it anyway?")], [sg.Button('Yes'),sg.Button('No')] ]
    window = sg.Window('Select',layout)
    event, values = window.Read()
    if event == "Yes":
        plt.imshow(mpimg.imread(im))
        plt.show()
    else:
        pass

net = torch.load('ensemble1.pt',map_location=torch.device('cpu'))
net = net.eval()
transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])

image_data = torchvision.datasets.ImageFolder(root='./data',transform=transform)

img_col = os.listdir('./data/AcrophobiaImages')
sig = nn.Sigmoid()

for i in range(0,len(img_col)):
    image_file = 'data/AcrophobiaImages/'+img_col[i]
    input_img = image_loader(transform,image_file)
    output = net(input_img)
    output = sig(output)
    output = output[0][0]
    __, index = torch.max(output,0)
    if index < 13 and AllPhobias[index] in ApplicablePhobias:
        AskUser(image_file,AllPhobias[index])
    elif index > 13 and AllPhobias[index-1] in ApplicablePhobias:
        AskUser(image_file,AllPhobias[index-1])