from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("./images/test_2.jpg")
imageB = cv2.imread("./images/test_1.jpg")
# imageA = cv2.imread("./images/left_01.png")
# imageB = cv2.imread("./images/right_01.png")

# 改变图像大小：
down_width = 500
down_height = 400
down_points = (down_width, down_height)
imageA = cv2.resize(imageA, down_points, interpolation=cv2.INTER_LINEAR)
imageB = cv2.resize(imageB, down_points, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Image A before train", imageA)
cv2.imshow("Image B before train", imageB)

print('开始执行...')
# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches line", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
