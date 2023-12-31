import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# 检查 GPU 还是 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 1、导入数据
train_data = dataset.MNIST(root='mnist', train=True, transform=transforms.ToTensor(), download=True)
test_data = dataset.MNIST(root='mnist', train=False, transform=transforms.ToTensor(), download=False)

# 2、batch size
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


# 3、定义网络：
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        # ?
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


cnn = CNN()


# 4、损失函数：

# 分类问题采用交叉熵的损失函数：
loss_func = torch.nn.CrossEntropyLoss()

# 5、优化器：
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# 6、训练过程：
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch is {}, ite is {}/{}, loss is {}'.format(epoch+1, i, len(train_data) // 64, loss.item()))

    # 7、测试
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        # 进行累加，计算平均 loss
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print('epoch is {}, accuracy is {}, loss is {}'.format(epoch+1, accuracy, loss_test))


# 8、保存模型
torch.save(cnn, 'model/mnist_model.pkl')
