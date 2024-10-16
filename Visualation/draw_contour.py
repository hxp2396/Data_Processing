#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 10:03
# @File : draw_contour.py
# @annotation:
# -------------------------批量在rgb原图上画出轮廓线---------------------------、
import cv2
import os
import numpy as np
def union_image_mask(image_path, mask_path, image_name, color = (255, 0, 255)):
    image = cv2.imread(image_path)
    chang = image.shape[1]
    kuan = image.shape[0]
    mask_2d = cv2.imread(mask_path,0)
    mask_2d = cv2.resize(mask_2d,(chang, kuan))
    coef = 255 if np.max(image)<3 else 1
    image = (image * coef).astype(np.float32)
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(image.shape)
    cv2.drawContours(image, contours, -1, color, 1)
    # cv2.imwrite(os.path.join('Add', image_name),image)
    cv2.imwrite(image_name,image)
def change_image(path):
    for img_name in os.listdir(path):
        print(img_name)
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img[img >0 ] = 255
        cv2.imwrite(img_path, img)
if __name__ == '__main__':
    #union_image_mask('1.jpg', '1_.jpg', 'masks.jpg')
    images = os.listdir('dataset/11') # 原图文件夹
    masks = os.listdir('output') # mask文件夹
    for image_name in images:
        image_path = os.path.join('dataset/11', image_name)
        mask_path = os.path.join('output', image_name)
        union_image_mask(image_path, mask_path,image_name)
    change_image('output')