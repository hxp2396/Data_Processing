#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 16:30
# @File : Fill_hole.py
# @annotation:根据边界线填充闭合区域
# ---------------------------------------------------------------根据边界线填充闭合区域-------------------------------------------
import os
import cv2
import numpy as np
from glob import glob
'''
图像说明：
图像为二值化图像，255白色为目标物，0黑色为背景
要填充白色目标物中的黑色空洞
'''
def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    print(im_in)
    # 复制 im_in 图像
    im_floodfill = im_in.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255);
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv
    # 保存结果
    cv2.imwrite(SavePath, im_out)

if __name__ == '__main__':
    filelist=glob('Ultra_HC/label/*.png')
    # print(filelist)
    save_dir='Ultra_HC/new_label'
    os.makedirs(save_dir,exist_ok=True)
    for file in filelist:
        FillHole(file,file.replace('label','new_label'))


