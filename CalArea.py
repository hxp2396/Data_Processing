#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  :
# @annotation :计算二维图像的面积和周长
# import PIL.Image
# import cv2 as cv
# import numpy as np
# def show(img):
#     img=PIL.Image.fromarray(img)
#     img.show()
#
# class ShapeAnalysis:
#     def __init__(self):
#         self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}
#
#     def analysis(self, frame):
#         h, w, ch = frame.shape
#         result = np.zeros((h, w, ch), dtype=np.uint8)
#         # 二值化图像
#         print("start to detect lines...\n")
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
#
#         # cv.imshow("input image", frame)
#         show(frame)
#
#         contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         for cnt in range(len(contours)):
#             # 提取与绘制轮廓
#             cv.drawContours(result, contours, cnt, (0, 255, 0), 2)
#
#             # 轮廓逼近
#             epsilon = 0.01 * cv.arcLength(contours[cnt], True)
#             approx = cv.approxPolyDP(contours[cnt], epsilon, True)
#
#             # 分析几何形状
#             corners = len(approx)
#             shape_type = ""
#             if corners == 3:
#                 count = self.shapes['triangle']
#                 count = count+1
#                 self.shapes['triangle'] = count
#                 shape_type = "三角形"
#             if corners == 4:
#                 count = self.shapes['rectangle']
#                 count = count + 1
#                 self.shapes['rectangle'] = count
#                 shape_type = "矩形"
#             if corners >= 10:
#                 count = self.shapes['circles']
#                 count = count + 1
#                 self.shapes['circles'] = count
#                 shape_type = "圆形"
#             if 4 < corners < 10:
#                 count = self.shapes['polygons']
#                 count = count + 1
#                 self.shapes['polygons'] = count
#                 shape_type = "多边形"
#
#
#             # 求解中心位置
#             mm = cv.moments(contours[cnt])
#             cx = int(mm['m10'] / mm['m00'])
#             cy = int(mm['m01'] / mm['m00'])
#             cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)
#
#             # 颜色分析
#             color = frame[cy][cx]
#             color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
#
#             # 计算面积与周长
#             p = cv.arcLength(contours[cnt], True)
#             area = cv.contourArea(contours[cnt])
#             print("周长: %.3f, 面积: %.3f 颜色: %s 形状: %s "% (p, area, color_str, shape_type))
#
#         show(self.draw_text_info(result))
#         # cv.imshow("Analysis Result", self.draw_text_info(result))
#         cv.imwrite("test-result.png", self.draw_text_info(result))
#         return self.shapes
#
#     def draw_text_info(self, image):
#         c1 = self.shapes['triangle']
#         c2 = self.shapes['rectangle']
#         c3 = self.shapes['polygons']
#         c4 = self.shapes['circles']
#         cv.putText(image, "triangle: "+str(c1), (10, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
#         cv.putText(image, "rectangle: " + str(c2), (10, 40), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
#         cv.putText(image, "polygons: " + str(c3), (10, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
#         cv.putText(image, "circles: " + str(c4), (10, 80), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
#         return image
#
#
# if __name__ == "__main__":
#     src = cv.imread("E:/DataCollection/1.png")
#     ld = ShapeAnalysis()
#     ld.analysis(src)
#     # cv.waitKey(0)
#     cv.destroyAllWindows()


import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 二值化
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.threshold第二个参数设定 红色通道阈值（阈值影响开闭运算效果）
    print("threshold value %s" % ret)
    # 显示
    # cv2.imshow("global_threshold_binary", binary)
    return binary


img = cv2.imread('28381.png')
# 对图像二值化处理
img = threshold_demo(img)

# OpenCV定义的结构矩形元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 腐蚀图像
eroded = cv2.erode(img, kernel)

# 膨胀图像
dilated = cv2.dilate(img, kernel)

# cv2.imshow("img", img)
# cv2.imshow("Eroded", eroded)
# cv2.imshow("Dilated", dilated)
# cv2.waitKey(0)

plt.subplot(131), plt.imshow(img,'gray') ,plt.title("原图")
plt.subplot(132), plt.imshow(dilated,'gray'), plt.title("膨胀")
plt.subplot(133), plt.imshow(eroded,'gray'), plt.title("腐蚀")
plt.show()

# 闭运算 迭代次数不同
closed1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
closed3 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
closed5 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
# 开运算
opened1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
opened3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
opened5 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=5)
# 梯度
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 显示如下腐蚀后的图像
# cv2.imshow("Close1", closed1)
# cv2.imshow("Close3", closed3)
# cv2.imshow("Open1", opened1)
# cv2.imshow("Open3", opened3)
# cv2.imshow("gradient", gradient)
# cv2.waitKey(0)

plt.subplot(241), plt.imshow(img,'gray') ,plt.title("原图")
plt.subplot(242), plt.imshow(opened1,'gray'), plt.title("开运算iterations=1")
plt.subplot(243), plt.imshow(opened3,'gray'), plt.title("开运算iterations=3")
plt.subplot(244), plt.imshow(opened5,'gray'), plt.title("开运算iterations=5")
plt.subplot(246), plt.imshow(closed1,'gray'), plt.title("闭运算iterations=1")
plt.subplot(247), plt.imshow(closed3,'gray'), plt.title("闭运算iterations=3")
plt.subplot(248), plt.imshow(closed5,'gray'), plt.title("闭运算iterations=5")
plt.show()


# 将两幅图像相减获得边；cv2.absdiff参数：(膨胀后的图像，腐蚀后的图像)
absdiff_img = cv2.absdiff(dilated, eroded);
result = cv2.bitwise_not(absdiff_img);

# cv2.imshow("absdiff_img", absdiff_img)
# cv2.imshow("result", result)
# cv2.waitKey(0)

plt.subplot(131), plt.imshow(img,'gray') ,plt.title("原图")
plt.subplot(132), plt.imshow(absdiff_img,'gray'), plt.title("腐蚀、膨胀两幅图像相减")
plt.subplot(133), plt.imshow(result,'gray'), plt.title("提取边缘")
plt.show()

