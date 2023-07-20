import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像读成灰度图：
img1 = cv2.imread('./images/box.png', 0)
img2 = cv2.imread('./images/box_in_scene.png', 0)


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv_show('img1', img1)

cv_show('img2', img2)

sift = cv2.SIFT_create()

# 计算每张图片的关键点和对应的特征向量：
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
# NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式
# 进行蛮力匹配:
bf = cv2.BFMatcher(crossCheck=True)

# ---------------- 1 对 1 的匹配 --------------------
matches = bf.match(des1, des2)
# 对距离进行排序:
matches = sorted(matches, key=lambda x: x.distance)
# 把匹配结果画出来
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
cv_show('img3', img3)


# ---------------- k对最佳匹配 ---------------------
bf = cv2.BFMatcher()
# 这里 k = 2 也就是 一个点 可以跟 2 个最近的点相对应
matches = bf.knnMatch(des1, des2, k=2)


good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv_show('img3 k match', img3)
