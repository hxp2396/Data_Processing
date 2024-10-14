#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 15:56
# @File : random_cut.py
# @annotation:随机裁剪图片
# -------------------------------------------------------------病理图像裁剪切片--------------------------------------------
from PIL import Image
import os
import numpy as np
def check_and_creat_dir(file_url):
    '''
    判断文件是否存在，文件路径不存在则创建文件夹
    :param file_url: 文件路径，包含文件名
    :return:
    '''
    file_gang_list = file_url.split('/')
    # print("okk",file_gang_list )
    if len(file_gang_list) > 1:
        [fname, fename] = os.path.split(file_url)
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            return None
        # 还可以直接创建空文件
    else:
        return None

"""
病理图像数据集扩充，
大尺寸病理图像剪切，包括：原图、上下翻转、左右翻转、旋转180、小角度随机旋转
"""
def cut_image_2part(image):
    width, height = image.size
    # print("image.size", image.size,image)
    item_width =299
    box_list = []
    for m in range(0,4):
        #创造剪切图片的随机起点
        x_start = np.random.randint(0, high=42, size=1, dtype='l').item()
        y_start = np.random.randint(0, high=101, size=1, dtype='l').item()
        for i in range(0, 1):
            for j in range(0, 2):
                box = (x_start+j * item_width, y_start+i * item_width, x_start+(j + 1) * item_width, y_start+(i + 1) * item_width)
                box_list.append(box)
                print( box)
                # plt.imshow(image.crop(box))
                # plt.show()
    image_list = [image.crop(box) for box in box_list]
    image_LR= image.transpose(Image.FLIP_LEFT_RIGHT)
    image_TB= image.transpose(Image.FLIP_TOP_BOTTOM)
    image_ROT= image.rotate(180)
    image_list = image_list + [image_LR.crop(box) for box in box_list]+\
                 [image_TB.crop(box) for box in box_list]+\
                 [image_ROT.crop(box) for box in box_list]
    cut_box_list = []
    x_start =20
    y_start =90
    for i in range(0, 1):
        for j in range(0, 2):
            box = (x_start+j * item_width, y_start+i * item_width, x_start+(j + 1) * item_width, y_start+(i + 1) * item_width)
            cut_box_list.append(box)
    print("cut_box_list",cut_box_list)
    rot1 = np.random.randint(0, high=20, size=1, dtype='l').item()
    rot2 = np.random.randint(-20, high=0,size=1, dtype='l').item()
    image_list = image_list + [image.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_LR.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_LR.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_TB.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_TB.rotate(rot2).crop(box) for box in cut_box_list]+\
                 [image_ROT.rotate(rot1).crop(box) for box in cut_box_list]+\
                 [image_ROT.rotate(rot2).crop(box) for box in cut_box_list]
    return image_list
def cut_image(image):
    width, height = image.size
    # print("image.size", image.size,image)
    item_width = 299
    box_list = []
    # # (left, upper, right, lower)
    for i in range(0, 1):  # 两重循环，生成9张图片基于原图的位置
        for j in range(0, 1):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)
            # plt.imshow(image.crop(box))
            # plt.show()
    image2 = image.rotate(100, expand=0)
    # plt.imshow(image2)
    # plt.show()
    image_list = [image.crop(box) for box in box_list]
    # if rotate:
    #     image_list = image_list + [image2.crop(box) for box in box_list]
    return image_list
# 保存在list输出到图片中
def save_images(file_path, image_list):
    index = 1
    for image in image_list:
        save_path="E:/DataCollection/XXX/cut/" + file_path+"_" + str(index) + '.tif'
        print(save_path)
        check_and_creat_dir(save_path)
        image.save(save_path)
        index += 1
if __name__ == '__main__':
    choice=1
    if choice==1:# -------------------随机裁剪病理图片----------------
        root ="E:/DataCollection/XXX/MonuSeg/Training"
        p = os.listdir(root)
        print(p)
        for file_path in p:
            file_root=os.path.join(root,file_path)
            p2 = os.listdir(file_root)
            for file_path2 in p2:
                image_path=os.path.join(file_root,file_path2)
                image = Image.open(image_path)
                image = image.resize((1024,1024))
                image_list = cut_image_2part(image)
                save_images(file_path + "/" + file_path2, image_list)