import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar10 import train_data_loader, test_data_loader
import os

# 判断 cpu GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 10
lr = 0.01
batch_size = 128

net = VGGNet().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# 学习率:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 5 个 epoch 之后,学习率变为原来的 0.9 倍

print('开始训练!!!')
for epoch in range(epoch_num):
    net.train()

    # 训练:
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算预测值:
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).sum()
        print('i :', i)
    print("train epoch is ", epoch)

    sum_loss = 0
    sum_correct = 0
    # 测试:
    for i, data in enumerate(test_data_loader):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        # 计算预测值:
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).sum()

        sum_loss += loss.item()
        sum_correct += correct.item()

    test_loss = sum_loss * 1.0 / len(test_data_loader)
    test_correct = sum_correct * 100.0 / len(test_data_loader) / batch_size

    print('test epoch is ', epoch, 'loss is: ', test_loss, 'test correct is: ', test_correct)

    # 模型存储:
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(net.state_dict(), 'models/{}.pth'.format(epoch + 1))
    scheduler.step()
    print('lr is', optimizer.state_dict()['param_groups'][0]['lr'])
