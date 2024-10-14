#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 16:07
# @File : contour_extraction.py
# @annotation:二值边缘提取
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
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
def contour_extraction(image_path):
    img = cv2.imread(image_path)
    # 对图像二值化处理
    img = threshold_demo(img)
    # OpenCV定义的结构矩形元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 腐蚀图像
    eroded = cv2.erode(img, kernel)
    # 膨胀图像
    dilated = cv2.dilate(img, kernel)
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title("原图")
    plt.subplot(132), plt.imshow(dilated, 'gray'), plt.title("膨胀")
    plt.subplot(133), plt.imshow(eroded, 'gray'), plt.title("腐蚀")
    plt.show()
    # 将两幅图像相减获得边；cv2.absdiff参数：(膨胀后的图像，腐蚀后的图像)
    absdiff_img = cv2.absdiff(dilated, eroded);
    result = cv2.bitwise_not(absdiff_img);
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title("原图")
    plt.subplot(132), plt.imshow(absdiff_img, 'gray'), plt.title("腐蚀、膨胀两幅图像相减")
    plt.subplot(133), plt.imshow(result, 'gray'), plt.title("提取边缘")
    plt.show()
    return result
def contour_extraction_2(origin_pic,contour_path):
    # --------------------------------------提取图片轮廓--------------------------------------
    img = cv2.imread(origin_pic)
    kernel = np.ones((3, 3), dtype=np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    ss = np.hstack((img, gradient))
    cv2.imwrite(contour_path, ss)
def Edge_Extract(root):
    img_root = os.path.join(root,'img_masks')			# 修改为保存图像的文件名
    edge_root = os.path.join(root,'img_edge')			# 结果输出文件
    if not os.path.exists(edge_root):
        os.mkdir(edge_root)
    file_names = os.listdir(img_root)
    img_name = []
    for name in file_names:
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))
    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0

if __name__ == '__main__':
    img_path='xxx.png'
    contour=contour_extraction(img_path)