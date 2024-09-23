import torch
from spacy.training.pretrain import pretrain
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

# 检测是否有可用的GPU，如果有的话，使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 5

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载你的数据集
data_dir = './dataset'  # 在这里填上你数据的路径
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train', 'val']}

# 获取类的数量
num_classes = len(image_datasets['train'].classes)

# 加载预训练的 VGG19 模型
model = models.vgg19(weights=None)
model = models.vgg19(pretrained=True)

# 将最后一层替换为新的线性层，它的输出大小为num_classes
model.classifier[6] = nn.Linear(4096, num_classes)

model = model.to(device)

if not next(model.parameters()).is_cuda:
    exit(0)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 训练和验证循环
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            # 将输入数据和标签发送至GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

        print('epoch: {} {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))

    scheduler.step()

print('Finished Training')
torch.save(model.state_dict(), 'model_VGG.pth')
