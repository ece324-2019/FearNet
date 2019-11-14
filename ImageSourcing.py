PhobiaSet = ("Heights","Flying","Open Spaces","Spiders","Lightning","Loneliness","Cancer","Enclosed Spaces","Dogs","Vomit","Crowds","Blood","Water","Snakes","Birds","Bacteria","Death","Holes","Needle Injection","Driving","None")

from PIL import Image

import torch 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.batchnorm as batchnorm
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import time

from torchsummary import summary

AllImages = torchvision.datasets.ImageFolder(root='.',transform=transforms.ToTensor())

Ch1Mean = 0.0
Ch1SD = 0.0
Ch2Mean = 0.0
Ch2SD = 0.0
Ch3Mean = 0.0
Ch3SD = 0.0

counter = 1
for img in AllImages:
    print("Image Counter: ",counter)
    Ch1Mean += img[0][0].mean()
    Ch1SD += img[0][0].std()
    Ch2Mean += img[0][1].mean()
    Ch2SD += img[0][1].std()
    Ch3Mean += img[0][2].mean()
    Ch3SD += img[0][2].std()
    counter += 1

Ch1Mean = Ch1Mean/len(AllImages)
Ch1SD = Ch1SD/len(AllImages)
Ch2Mean = Ch2Mean/len(AllImages)
Ch2SD = Ch2SD/len(AllImages)
Ch3Mean = Ch3Mean/len(AllImages)
Ch3SD = Ch3SD/len(AllImages)

print(Ch1Mean, "\n", Ch1SD, "\n", Ch2Mean, "\n", Ch2SD, "\n", Ch3Mean, "\n", Ch3SD)

transform = transforms.Compose(
    [Image.open().resize((64,64)),
     transforms.ToTensor(),transforms.Normalize(
        (Ch1Mean.item(),Ch2Mean.item(),Ch3Mean.item()),
        (Ch1SD.item(),Ch2SD.item(),Ch3SD.item())
     )
    ]
)

AllImages = torchvision.datasets.ImageFolder(root='.',transform=transform)

AllImagesLoader = torch.utils.data.DataLoader(AllImages, batch_size=4, shuffle=True, num_workers=4)

dataiter = iter(AllImagesLoader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))