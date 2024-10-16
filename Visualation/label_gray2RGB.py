#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 9:46
# @File : label_gray2RGB.py
# @annotation:
def grey_rgb(imgpath,saveroot='/',classnum=2,colorlist=[]):
    label = Image.open(imgpath)
    filename=os.path.split(imgpath)[-1]
    label = label.convert("RGBA")
    width = label.size[0]  # 长度
    height = label.size[1]  # 宽度
    name = os.path.join(saveroot,filename)
    print(name)
    for i in range(0, width):  # 遍历所有长度的点
        for j in range(0, height):  # 遍历所有宽度的点
            data = label.getpixel((i, j))  # i,j表示像素点
            for k in range(0,classnum):
                if (data[0] == k and data[1] == k and data[2] == k):
                    label.putpixel((i, j), colorlist[k])  # 颜色改变
    label.save(name)
if __name__ == '__main__':
    colorlist=[(255,255,255),(255,0,0),(0,255,0),(0, 0, 255), (139, 0, 0), (0, 0, 139), (139, 0, 139), (144, 238, 144),
                (0, 139, 139), (155, 48, 288)]
    imapath='ncase0005_slice035.png'
    grey_rgb(imapath,'U/',9,colorlist)