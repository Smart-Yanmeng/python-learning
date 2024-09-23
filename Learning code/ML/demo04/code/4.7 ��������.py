import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from PIL import Image
import cv2


class MYNET(nn.Module):
    def __init__(self):
        super(MYNET, self).__init__()
        # 3个参数分别是输入通道数，输出通道数，卷积核大小
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 泛化测试
# 测试图像处理
transform = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.Grayscale(), transforms.ToTensor()]
)

# 加载模型
model = MYNET()
model.load_state_dict(torch.load("../tmp/model.pkl"))
model.eval()

# 输入图像并预测
img = Image.open("data/test/00007/`.png")
img = transform(img)
img = img.view(1, 1, 64, 64)
output = model(img)
_, prediction = torch.max(output, 1)
prediction = prediction.numpy()[0]
print(prediction)

