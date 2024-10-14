#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 16:21
# @File : 2d_pic_trans.py
# @annotation:
from PIL import Image
import os
from glob import glob
import cv2
from PIL import Image
def pic_trans(path,format):
    filelist = glob(path+'/*.tif')
    for name in filelist:
        img = cv2.imread(name)
        filename = os.path.split(name)[-1].replace('tif',format)
        cv2.imwrite(filename, img)
        print(filename)
if __name__ == '__main__':
    path=r'C:\Users\hxp\Desktop\DataCollection'
    format = 'jpg'
    pic_trans(path,format)