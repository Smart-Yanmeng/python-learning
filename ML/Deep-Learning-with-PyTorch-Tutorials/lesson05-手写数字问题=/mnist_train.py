import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

n_epochs = 3
batch_size_train = 64
batch_size_test = 512
learning_rate = 0.00788
momentum = 0.966
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # 卷积层 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 卷积层 2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 展平
        x = x.view(-1, 320)
        # 全连接层 1
        x = F.relu(self.fc1(x))
        # 丢弃层
        x = F.dropout(x, training=self.training)
        # 全连接层 2
        x = self.fc2(x)
        # 对数
        return F.log_softmax(x, dim=1)


# 初始化网络和优化器
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

for name in network.state_dict():
    print(network.state_dict()[name]
          )

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # 手动将梯度设置为零
        optimizer.zero_grad()

        output = network(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')


def main():
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    torch.save(network.state_dict(), 'mnist_train_network.pth')


if __name__ == '__main__':
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        main()
