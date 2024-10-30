#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/30 16:09
# @File : main.py
# @annotation:主要调用各种函数实现数据处理的目的
import os
import cv2
from glob import glob

import numpy as np

src='dataset/PNG/folder5/ADC'
files=glob(os.path.join(src,'*.png'))
for file in files:
    img = cv2.imread(file)
    if len(np.unique(img))==1:
        os.remove(file)