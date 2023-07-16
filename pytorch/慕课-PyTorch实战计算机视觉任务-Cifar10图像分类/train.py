import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet, net_name
# from resnet import resNet, net_name
# from mobilenetv1 import mobilenet, net_name
# from inceptionMolule import InceptionNetSmall, net_name
# from pre_resnet import pytorch_resnet18, net_name
from load_cifar10 import train_data_loader, test_data_loader
import os
from torch.utils.tensorboard import SummaryWriter

# 判断 cpu GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 10
lr = 0.01
batch_size = 128

# net = resNet().to(device)
# net = mobilenet().to(device)
# net = InceptionNetSmall().to(device)
# net = pytorch_resnet18().to(device)
net = VGGNet().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


# 学习率:
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 5 个 epoch 之后,学习率变为原来的 0.9 倍

if not os.path.exists(r'C:\Users\LiuHao\Desktop\python-data\log'):
    os.mkdir(r'C:\Users\LiuHao\Desktop\python-data\log')

writer = SummaryWriter(r'C:\Users\LiuHao\Desktop\python-data\log')

print('开始训练!!!')
step_n = 0
for epoch in range(epoch_num):
    net.train()

    # 训练:
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        # 优化器梯度为 0
        optimizer.zero_grad()
        # 进行反向传播
        loss.backward()
        # 完成参数更新
        optimizer.step()

        # 计算预测值:
        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(labels.data).sum()
        # tensorboard 展示：
        writer.add_scalar("train loss", loss.item(), global_step=step_n)
        writer.add_scalar("train correct", 100.0 * correct.item() / batch_size, global_step=step_n)
        im = torchvision.utils.make_grid(inputs)
        writer.add_image("train im", im, global_step=step_n)

        step_n += 1

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
        # tensorboard 展示：
        im = torchvision.utils.make_grid(inputs)
        writer.add_image("test im", im, global_step=step_n)

    test_loss = sum_loss * 1.0 / len(test_data_loader)
    test_correct = sum_correct * 100.0 / len(test_data_loader) / batch_size

    writer.add_scalar("test loss", test_loss, global_step=epoch+1)
    writer.add_scalar("test correct", test_correct, global_step=epoch+1)

    print('test epoch is ', epoch, 'loss is: ', test_loss, 'test correct is: ', test_correct)

    # 模型存储:
    if not os.path.exists('models/{}'.format(net_name)):
        os.mkdir('models/{}'.format(net_name))
    torch.save(net.state_dict(), 'models/{}/{}.pth'.format(net_name, epoch + 1))
    # 在每个 epoch 之后对学习率进行更新
    scheduler.step()
    print('lr is', optimizer.state_dict()['param_groups'][0]['lr'])


writer.close()
