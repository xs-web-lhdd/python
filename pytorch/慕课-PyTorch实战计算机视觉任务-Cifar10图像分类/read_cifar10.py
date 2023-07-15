import os
import cv2
import numpy as np
import glob


# cifar10 官网提供对下载文件的解码方式:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 定义 10 个类别:
label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# 将文件读取出来:
train_list = glob.glob(r'C:\Users\LiuHao\Desktop\python-data\cifar-10-batches-py\train_batch_*')

# 定义图像存放路径:
save_path = r'C:\Users\LiuHao\Desktop\python-data\cifar-10-batches-py\train'


# print("train_list: {}".format(train_list))

for file_path in train_list:
    file_data = unpickle(file_path)
    # print("打印字典的key: ")
    # print(file_data.keys())

    # 拿到图片的 索引值 和 数据值(向量)
    for im_idx, im_data in enumerate(file_data[b'data']):
        im_label = file_data[b'labels'][im_idx]  # 这里值为 0 1 2 3 ... 9
        im_name = file_data[b'filenames'][im_idx]
        # 将图片进行存储:

        # 拿到图片所属类别的名字
        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        # 将通道进行更换,将 3 这个通道移到最后,变为 32 × 32 × 3:
        im_data = np.transpose(im_data, (1, 2, 0))

        # 将图片进行可视化展示
        # cv2.imshow('im_data', im_data)
        # cv2.waitKey(0)  # 只有按下空格的时候才会跳转到下一张

        if not os.path.exists("{}/{}".format(save_path, im_label_name)):
            os.mkdir("{}/{}".format(save_path, im_label_name))

        # 将图片写入文件夹:
        cv2.imwrite("{}/{}/{}".format(save_path, im_label_name, im_name.decode('utf-8')), im_data)
