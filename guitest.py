from PIL import Image
import torch
from torchvision.transforms import transforms
from torch import nn

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

data_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('ensemble1.pt')
model = model.to(device)
model.eval()
img = image_loader(data_transforms,'./acrotest.jpg').to(device)

sig = nn.Sigmoid()
print(sig(model(img.detach())).detach())