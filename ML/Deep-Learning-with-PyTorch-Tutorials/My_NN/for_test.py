import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch import nn

from torch.autograd import Variable
import os
from PIL import Image
from torchvision.models import vgg19

classes = (
    'n01532829', 'n01704323', 'n01749939', 'n01770081', 'n01855672', 'n01910747', 'n01930112', 'n02089867', 'n02138441',
    'n02457408', 'n02606052', 'n02687172', 'n03047690', 'n03146219', 'n03773504')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ResNet-18
# model = torch.load("models/model_ResNet.pth")

# VGG-19
model = vgg19(pretrained=False)
model.classifier[6] = nn.Linear(4096, 15)
model_state_dict = torch.load("models/model_VGG.pth", map_location=DEVICE)
model.load_state_dict(model_state_dict)

for name in model.state_dict():
    print(model.state_dict()[name])

model.to(DEVICE)
model.eval()

path = './dataset/test/'
testList = os.listdir(path)

aaawaaa = 0

for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)

    # Predict
    _, pred = torch.max(out.data, 1)

    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))

    if file[0:9] == classes[pred.data.item()]:
        aaawaaa += 1

print("acc: {:.6f}".format(aaawaaa / len(testList)))
