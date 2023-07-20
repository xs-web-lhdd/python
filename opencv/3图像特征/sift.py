import cv2
import numpy as np

# 读取图像：
img = cv2.imread('./images/box.png')
# 转化成灰度图：
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行实例化：
sift = cv2.SIFT_create()
# kp keypoint就是关键点
kp = sift.detect(gray, None)
# 绘制关键点：
img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('drawKeypoints：', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算特征： kp 是关键点，des 是每一个关键点对应的特征
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape)

