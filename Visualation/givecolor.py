#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  :
# @annotation :用于对二维图像进行连通域分析并赋予不同的颜色
# -----------------------------------------给予每一个连通区域添加不同的颜色------------------------------------
import cv2
from skimage import measure, color
import numpy as np
import cv2
import random


def givecolor_1(path):
    img = cv2.imread(path)
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_temp = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
    labels = measure.label(img_temp)
    dst = color.label2rgb(labels, bg_label=0)  # bg_label=0要有，不然会有警告
    cv2.imshow("666", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def givecolor_2(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    Rows, Cols = img.shape
    color = []
    color.append([0, 0, 0])  # 背景色
    for i in range(num):
        color.append([
            random.randint(0, 32767) % 256,
            random.randint(0, 32767) % 256,
            random.randint(0, 32767) % 256
        ])
    src_color = np.zeros((Rows, Cols, 3), dtype=np.uint8)
    for x in range(Rows):
        for y in range(Cols):
            label = labels[x][y]  # 图像总共有 num 个连通块, labels 会告诉你每一个坐标属于哪一个连通块
            # color 总共分配了 num + 1 种随机颜色, 每一个连通块都能分到一个随机色
            src_color[x][y] = color[label]
    cv2.imshow("Perpesctive transform", src_color)
    cv2.waitKey()


if __name__ == '__main__':
    path = "14800_2140.png"
    givecolor_1(path)
    givecolor_2(path)
"""
虽然python 3 使用统一编码解决了中文字符串的问题, 但在使用opencv中imread函数读取中文路径图像文件时仍会报错
此时可借助于numpy 先将文件数据读取出来, 然后使用opencv中imdecode函数将其解码成图像数据。此方法对python 2 和3均使用。
"""
