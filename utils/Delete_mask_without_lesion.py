#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 15:58
# @File : Delete_mask_without_lesion.py
# @annotation:
#     # ---------------------------------删除无病灶图片
from tqdm import tqdm
from glob import glob
import cv2
import os
import numpy as np
def delete_pic_no_lesion(rootpath):# rootpath=r'E:\DataCollection\spleen\masks'
    files=glob(rootpath+'/*.jpg')
    # print(files)
    for file in files:
        array=cv2.imread(file)
        if len(np.unique(array))==1:
            os.remove(file)
        print(np.unique(array))
def binary_mask(mask_path):
    files = glob(mask_path+'/*.bmp')
    with tqdm(total=len(files)) as bar: # total表示预期的迭代次数
        for file in files:
            array=cv2.imread(file,0)
            h,w=array.shape
            print(np.unique(array))
            array[array>0]=255
            array[array < 254] = 0
            cv2.imwrite(file,array)
            bar.update()