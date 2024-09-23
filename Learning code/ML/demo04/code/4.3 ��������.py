# 代码4-4 
# 构建卷积神经网络
# 构建网络
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
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
    
# 查看网络结构
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MYNET().to(device)
summary(model, (1, 64, 64))


