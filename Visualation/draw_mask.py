#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:28
# @File : draw_mask.py
# @annotation:
# ---------------------------------------------------------将标签以mask的形式画在原图-----------------------------------
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image
def draw_mask_1(img_path, mask_path):
    image = mpimg.imread(img_path)
    plt.imshow(image)
    plt.axis('off') # 不显示坐标轴
    plt.show()
    image.flags.writeable = True  # 将数组改为读写模式`
    Image.fromarray(np.uint8(image))
    masks = mpimg.imread(mask_path)
    plt.imshow(masks)
    plt.axis('off') # 不显示坐标轴
    plt.show()
    print(masks.shape)
    image[:,:,:][masks[:,:,:]>0] = 255
    img=Image.fromarray(image)
    img.save('sert.png')
    cv2.imwrite('test.png',image)
def _draw_to_overlay_image(labimgs, srcimgs,save_path,h,w):
    num_examples = h * w####图像的大小
    labimg = cv2.imread(labimgs)
    srcimg = cv2.imread(srcimgs)
    srcimg = np.reshape(srcimg, [num_examples, 3]) # //原图
    labimg = np.reshape(labimg, [num_examples, 3])  #//标签图
    label_mage=np.zeros([num_examples, 3], np.uint8) #//定义融合后的新图矩阵
    colors=[[255, 193, 37],[0,0,250]]   #//每一个类别对应一种颜色
    lab_pix = [0, 255]  # //每个类别对应的像素值
    # colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 0], [0, 0, 139], [139, 0, 139], [144, 238, 144],
    #           [0, 139, 139], [155, 48, 288], [255, 62, 150], [255, 165, 0],
    #           [255, 211, 155], [255, 193, 37], [255, 255, 0], [192, 255, 62], [0, 255, 255], [153, 50, 204],
    #           [255, 162, 0], [50, 205, 50], [0, 255, 255],
    #           [47, 79, 79], [119, 136, 153], [25, 25, 112], [123, 104, 238], [135, 206, 250], [0, 100, 0],
    #           [173, 255, 47], [188, 143, 143], [250, 128, 114],
    #           [205, 102, 29], [205, 51, 51], [205, 16, 118]]  # //每一个类别对应一种颜色
    # lab_pix = [33, 37, 35, 36, 34, 165, 166, 167, 50, 66, 164, 40, 163, 39, 38, 168, 49, 65, 162, 161, 67, 81, 82, 83,
    #            84, 85, 86, 97, 98, 99, 100, 113] #// 每个类别对应的像素值
    yuv_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [-0.14714119, -0.28886916, 0.43601035],
                             [0.61497538, -0.51496512, -0.10001026]])   #//YUV和RGB空间转换的参数矩阵
    rgb_from_yuv = np.linalg.inv(yuv_from_rgb)     #//求转置矩阵
    for i in range(0,num_examples):
        for a,b in enumerate(lab_pix):
            if labimg[i][0] == b:
                label_mage[i] = colors[a]    #//比对标签图中的像素值所属类别，然后下面进行颜色空间的转换
        Y = srcimg[i].dot(yuv_from_rgb[0].T.copy())
        U = label_mage[i].dot(yuv_from_rgb[1].T.copy())
        V = label_mage[i].dot(yuv_from_rgb[2].T.copy())
        rgb = np.array([Y, U, V]).dot(rgb_from_yuv.T.copy())
        if rgb[0] > 255: rgb[0] = 255   #//超出像素值255的全部设置为255，下同
        if rgb[1] > 255: rgb[1] = 255
        if rgb[2] > 255: rgb[2] = 255
        if rgb[0] < 0: rgb[0] = 0
        if rgb[1] < 0: rgb[1] = 0
        if rgb[2] < 0: rgb[2] = 0
        label_mage[i] = rgb
    rimg = np.reshape(label_mage, [h, w, 3])#图像大小
    Image.fromarray(rimg).save(save_path)
    cv2.imwrite(save_path, rimg)
def draw_mask_2d(srcimg, mask,save_path=None):
    img = Image.open(srcimg)
    img2 = Image.open(mask)
    merge = Image.blend(img, img2, 0.3)
    merge.save(save_path)

if __name__ == '__main__':
    pass