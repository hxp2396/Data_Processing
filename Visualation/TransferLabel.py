#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 15:58
# @File : Delete_mask_without_lesion.py
# @annotation:将多类标签彩色形式展示
import os
from glob import glob
import cv2
import numpy as np
from matplotlib import image as mpimg
from skimage.color import gray2rgb
from PIL import Image
# ------------------------------多类别分割图以不同颜色表示------------------
attr_colors = {
'0':(255, 0, 0),
'1':(0, 255, 0),
'2':(0, 0, 255),
'3':(139, 0, 0),
'4':(0, 0, 139),
'5':(139, 0, 139),
'6':(144, 238, 144),
'7':(0, 139, 139),
'8':(155, 48, 288),
'9':(255, 62, 150),
'10':(255, 165, 0),
'11':(255, 211, 155),
'12':(255, 193, 37),
'13':(255, 255, 0),
}
def put_predict_image(origin_image_np, test_mask, attr, alpha):
    '''
    将predict图片以apha透明度覆盖到origin图片中
    :param origin_image:
    :param predict_image:
    :param RGB:
    :param alpha:
    :return:
    '''
    test_mask_RGB = Image.fromarray(test_mask.astype('uint8')).convert("RGB") # 将原始二值化图像转换成RGB
    test_mask_np = np.asarray(test_mask_RGB,dtype=np.int) # 将二值化图像转换成三维数组
    height, width, channels = test_mask_np.shape  # 获得图片的三个纬度
    # 转换预测图像的颜色
    origin_image_np.flags.writeable=True
    test_mask_np.flags.writeable = True
    for row in range(height):
        for col in range(width):
            # 上色，这里将mask图像中白色部分转换为我们想要的颜色，
            if test_mask_np[row, col, 0] == 255 and test_mask_np[row, col, 1] == 255 and test_mask_np[row, col, 2] == 255:
                test_mask_np[row, col, 0] = attr_colors[attr][0]
                test_mask_np[row, col, 1] = attr_colors[attr][1]
                test_mask_np[row, col, 2] = attr_colors[attr][2]
            # 这里对我们关心的白色区域，将这一步分像素按照比例相加。
            if test_mask_np[row, col, 0] != 0 or test_mask_np[row,col, 1] != 0 or test_mask_np[row, col, 2] != 0:
                origin_image_np[row,col,0] = alpha*origin_image_np[row,col,0] + (1-alpha)*test_mask_np[row, col, 0]
                origin_image_np[row,col,1] = alpha*origin_image_np[row,col,1] + (1-alpha)*test_mask_np[row, col, 1]
                origin_image_np[row,col,2] = alpha*origin_image_np[row,col,2] + (1-alpha)*test_mask_np[row, col, 2]
    img = Image.fromarray(origin_image_np)
    # img.save('test.png')
    return origin_image_np
def givecolor(labepath,savepath=None):
    filename=os.path.split(labepath)[-1]
    label=labepath
    labe=cv2.imread(label,0)
    img=np.zeros_like(labe)
    cv2.imwrite('temp.jpg',img)
    origin = mpimg.imread('temp.jpg')####不能为png格式
    if len(origin.shape)!=3:
        origin=gray2rgb(origin)
    pixellist=np.unique(labe)
    for iters in pixellist:
        if iters==0:
            continue
        img=np.zeros_like(labe)
        # img[pred==iters]=255
        # img[pred<255] =0
        h,w=img.shape
        for i in range(0,h):
            for j in range(0,w):
                if labe[i][j]==iters:
                    img[i][j]=255
                else:
                    img[i][j]=0
        # print(origin.shape)
        # print(img.shape)
        # print(numpy.unique(img))
        origin=put_predict_image(origin, img, str(iters), 0.4)
    new_img = Image.fromarray(origin)
    if savepath is not None:
        save_path=os.path.join(savepath,filename)
    else:
        save_path=labepath
    new_img.save(save_path)
    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')

if __name__ == '__main__':
    choice=2
    if choice==2:###将多类标签以不同的颜色展示
        label_path = 'U/Transfer'
        filelist = glob(label_path + '/*.jpg')
        for file in filelist:
            givecolor(file)
