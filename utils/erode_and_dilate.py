#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 16:52
# @File : erode_and_dilate.py
# @annotation:# -----------------------------------------膨胀腐蚀----------------------------------------------------------
import cv2 as cv
def erode_demo(image):
    # print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))  # 定义结构元素的形状和大小
    dst = cv.erode(binary, kernel)  # 腐蚀操作
    cv.imshow("erode_demo", dst)
def dilate_demo(image):
    # print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
    dst = cv.dilate(binary, kernel)  # 膨胀操作
    cv.imshow("dilate_demo", dst)
if __name__ == "__main__":
    src = cv.imread("Masks/img_edge/10.png")
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)
    erode_demo(src)
    dilate_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
