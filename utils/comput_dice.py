#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:12
# @File : comput_dice.py
# @annotation:# -------------------#计算Dice-----------------------------------------------------------------
from glob import glob
import cv2
import numpy as np
def cal_dice(seg, gt, classes=9, background_id=0):
    channel_dice = []
    a=np.array(np.unique(seg))
    b=np.array(np.unique(gt))
    c=(a==b)
    if len(np.unique(seg))==len(np.unique(gt)) and len(np.unique(seg))==1 and c:
        channel_dice.append(1)
    else:
        for i in range(classes):
            if i == background_id:
                continue
            if i not in np.unique(seg) and i not in np.unique(gt):
                continue
            else:
                cond = i ** 2
                # 计算相交部分
                inter = len(np.where(seg * gt == cond)[0])
                total_pix = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0])
                if total_pix == 0:
                    dice = 0
                else:
                    dice = (2 * inter) / total_pix
            # print(dice)
            channel_dice.append(dice)
    res = np.array(channel_dice).mean()
    print(res)
    return res

if __name__ == '__main__':
    filelist=glob(r'E:\DataCollection\test_mask\*.png')
    dice=0.
    for name in filelist:
        name1=name
        name2=name.replace('test_mask','pred')
        img1=cv2.imread(name1,0)
        img1=cv2.resize(img1,(224,224),interpolation=cv2.INTER_NEAREST)
        img2=cv2.imread(name2,0)
        dice+=cal_dice(img1,img2)
    print(dice/1568)
