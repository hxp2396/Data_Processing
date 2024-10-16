#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/16 10:32
# @File : FFT.py
# @annotation:

import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
im_gray = cv2.imread("image/000008.png", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
cv2.imwrite('ndvi_color8.jpg', im_color)
img = cv.imread("image/000092.png")
img = np.double(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
m, n = img.shape
rL = 0.5
rH = 2
c =2
d0 = 20
A1 = np.log(img+1)
FI = np.fft.fft2(A1)
n1 = np.floor(m/2)
n2 = np.floor(n/2)
D = np.zeros((m, n))
H = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        D[i, j] = ((i-n1)**2 + (j-n2)**2)
        H[i, j] = (rH-rL) * (np.exp(c * (-D[i, j] / (d0**2))))+rL
A2 = np.fft.ifft2(H*FI)
A3 = np.real(np.exp(A2))
plt.figure()
plt.imshow(img, cmap='gray')
plt.figure()
plt.imshow(A3, cmap='gray')
plt.show()