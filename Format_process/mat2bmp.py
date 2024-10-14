#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  :
# @annotation :这个文件的目的是将mat转为bmp图像
import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def MatrixToImage(data):# 数据矩阵转图片的函数
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
def mat2bmp(dataFile):
    data = scio.loadmat(dataFile)
    print(data)
    print(type(data))
    # 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
    a=data['inst_map']# 取出需要的数据矩阵
    new_im = MatrixToImage(a)
    plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
    new_im.show()
    new_im.save('data_2.bmp') # 保存图片
if __name__ == '__main__':
    dataFile = r'cpm15/Labels/image_00.mat' # 单个的mat文件
    mat2bmp(dataFile)