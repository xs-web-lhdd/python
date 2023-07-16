## Cifar 10/100 数据集介绍 & 下载
- Cifar 10/100：
  - 8000万个微小图像数据集的子集
  - 由 Alex Krizhevsky，Vinod Nair，Geoffrey Hinton收集
  - 官网链接：http://www.cs.toronto.edu/~kriz/cifar.html

- 数据下载压缩包打开后，对应 data_batch1-5，每个 batch 对应 1 万张图片，test_batch 是测试集，对应 1 万张图片

## 将下载的 Cifar10 数据集转成图片存储
- 将数据集进行处理为一张张的图片存储在文件夹内,代码: read_cifar.py

## 自定义数据加载对 Cifar10 数据进行加载
- 将图片数据加载出来, 训练集 50000 张, 测试集 10000 张,代码: load_cifar.py

## 搭建网络模型:
- 搭建一个类似 VGG 的网络结构, 代码: vggnet.py

## 训练:
- 开始训练:

## 分类模型优化思路

- 调参技巧：
  - backbone（主干网络）
  - 过拟合问题
  - 学习率调整
  - 优化函数
  - 数据增强
    - 亮度、对比度、饱和度
    - 裁剪、旋转




