#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 10:14
# @File : test_CalArea.py
# @annotation:测试函数
from Statistic.CalArea import *
from Statistic.CalArea import AreaEstimator
from PIL import Image
if __name__ == "__main__":
    area_estimator = AreaEstimator(filename='./1345.png',feet_per_pixel=72/138,color_range=(200,255))
    # 若不指定参数，则等价于下方
    # area_estimator = AreaEstimator(filename='PartyRockFire.jpeg',  # 要计算面积的图像路径
    #                                feet_per_pixel=8188/215,  # 像素值到英寸的转换，要根据图像的DPI才能计算
    #                                color_range=((0, 0, 118), (100, 100, 255)),  # 计算面积的像素值范围，
    #                                default_area_color=(147, 20, 255))  # 用什么颜色显示计算面积的那部分
    # 现在我们计算面积（单位为平方英尺），然后打印出保留三位有效数字的值
    A = area_estimator.get_area()
    print(A)
    # You can also get the area in pixels
    A_px = area_estimator.get_area(return_pixels=True)
    print(f'\nThe area of the Party Rock Fire is approximately {round(A, 3)} square feet (or {A_px} pixels)')
    # display the two images, then press any key to continue
    area_estimator.show_images()
    # you can can also get the area by adding over the columns instead of rows
    A_px_by_col = area_estimator.get_area(return_pixels=True, by_columns=True, fill_color=(255, 20, 147))  # Blue?
    A_by_col = A_px_by_col*area_estimator.px_size
    print(f'\nThe area as estimated by column-sum: {round(A_by_col, 3)} square feet ({A_px_by_col} px)')
    area_estimator.show_images()

    # and finally we can see the difference between the areas by not clearing the selected image
    area_estimator.get_area()  # horizontal sum
    area_estimator.area_color = (51, 244, 244)  # yellow
    area_estimator.get_area(by_columns=True)  # vertical sum
    area_estimator.show_images()

    print(f'\nthe average of the two methods is: {round(((A + A_by_col) / 2), -6)} square feet')
