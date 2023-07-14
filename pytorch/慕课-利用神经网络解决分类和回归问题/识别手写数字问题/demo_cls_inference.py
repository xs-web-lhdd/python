import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import cv2



# 1、导入测试数据
test_data = dataset.MNIST(root='mnist', train=False, transform=transforms.ToTensor(), download=False)

# 2、batch size
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# 3、导入训练好的模型
cnn = torch.load('model/mnist_model.pkl')

# 4、测试
accuracy = 0

for i, (images, labels) in enumerate(test_loader):
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()

    images = images.numpy()
    labels = labels.numpy()
    pred = pred.numpy()

    # batchsize * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]

        # 图片通道处理一下：
        im_data = im_data.transpose(1, 2, 0)

        print('label', im_label)
        print('pred', im_pred)
        # 进行图片展示：
        cv2.imshow('imdata', im_data)
        cv2.waitKey(0)

accuracy = accuracy / len(test_data)

print('accuracy is {}'.format(accuracy))

