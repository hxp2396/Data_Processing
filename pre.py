# import cv2
# import numpy as np
# from pywt import dwt2, idwt2
#
# # 读取灰度图
# img = cv2.imread('image/000004.png', 0)
#
# # 对img进行haar小波变换：
# cA, (cH, cV, cD) = dwt2(img, 'haar')
#
# # 小波变换之后，低频分量对应的图像：
# cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))
# # 小波变换之后，水平方向高频分量对应的图像：
# cv2.imwrite('lena_h.png', np.uint8(cH / np.max(cH) * 255))
# # 小波变换之后，垂直平方向高频分量对应的图像：
# cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
# # 小波变换之后，对角线方向高频分量对应的图像：
# cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))
#
# # 根据小波系数重构回去的图像
# rimg = idwt2((cA, (cH, cV, cD)), 'haar')
# cv2.imwrite('rimg.png', np.uint8(rimg))
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
from torch import einsum


def uniq(a):
    return set(torch.unique(a.cpu()).numpy())
def sset(a, sub):
    return uniq(a).issubset(sub)
def one_hot(t, axis=1) :
    return simplex(t, axis) and sset(t, [0, 1])
def class2one_hot(seg, C) :
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
#     assert one_hot(res)
    return res
def simplex(t, axis=1):
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
def one_hot2dist(seg) :
    assert one_hot(torch.Tensor(seg), axis=0)
    C= len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc= kwargs["idc"]
        print(self.__class__.__name__,kwargs)

    def __call__(self, probs, dist_maps, _) :
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
# import cv2
#
# im_gray = cv2.imread("image/000008.png", cv2.IMREAD_GRAYSCALE)
# im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
# cv2.imwrite('ndvi_color8.jpg', im_color)

# import cv2 as cv
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import ndimage
# from skimage import data, util, color
# import math
# img = cv.imread("image/000092.png")
# img = np.double(cv.cvtColor(img, cv.COLOR_RGB2GRAY))
# m, n = img.shape
# rL = 0.5
# rH = 2
# c =2
# d0 = 20
# A1 = np.log(img+1)
# FI = np.fft.fft2(A1)
# n1 = np.floor(m/2)
# n2 = np.floor(n/2)
# D = np.zeros((m, n))
# H = np.zeros((m, n))
# for i in range(m):
#     for j in range(n):
#         D[i, j] = ((i-n1)**2 + (j-n2)**2)
#         H[i, j] = (rH-rL) * (np.exp(c * (-D[i, j] / (d0**2))))+rL
# A2 = np.fft.ifft2(H*FI)
# A3 = np.real(np.exp(A2))
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.figure()
# plt.imshow(A3, cmap='gray')
# plt.show()

#  -*- coding: utf-8 -*-
import cv2
import os

def Edge_Extract(root):
    img_root = os.path.join(root,'img_masks')			# 修改为保存图像的文件名
    edge_root = os.path.join(root,'img_edge')			# 结果输出文件

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image,0)
        cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        index += 1
    return 0

#
# if __name__ == '__main__':
#     root = 'Masks/'	# 修改为你对应的文件路径
#     Edge_Extract(root)
# -----------------------------------------膨胀腐蚀----------------------------------------------------------
# import cv2 as cv
# import numpy as np
#
#
# def erode_demo(image):
#     # print(image.shape)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     # cv.imshow("binary", binary)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))  # 定义结构元素的形状和大小
#     dst = cv.erode(binary, kernel)  # 腐蚀操作
#     cv.imshow("erode_demo", dst)
#
#
# def dilate_demo(image):
#     # print(image.shape)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#     # cv.imshow("binary", binary)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义结构元素的形状和大小
#     dst = cv.dilate(binary, kernel)  # 膨胀操作
#     cv.imshow("dilate_demo", dst)
#
#
# src = cv.imread("Masks/img_edge/10.png")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# erode_demo(src)
# dilate_demo(src)
#
# cv.waitKey(0)
#
# cv.destroyAllWindows()

import cv2
import numpy as np

'''
图像说明：
图像为二值化图像，255白色为目标物，0黑色为背景
要填充白色目标物中的黑色空洞
'''


def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    print(im_in)

    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 255);

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    cv2.imwrite(SavePath, im_out)

# FillHole('Masks/img_edge/5.png','teast.png')
#--------------------------------------提取图片轮廓--------------------------------------
from PIL.ImageShow import show

# img = cv2.imread(r'C:\Users\hxp\Desktop\img_hence\Masks\img_masks\5.png')
#
# kernel = np.ones((5, 5), dtype=np.uint8)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, 1)
# ss = np.hstack((img, opening))
#
#
# cv2.imwrite('seg_smooth.png', opening)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
#
#
# img = cv2.imread(r'C:\Users\hxp\Desktop\img_hence\Masks\img_masks\5.png')
# kernel = np.ones((3, 3), dtype=np.uint8)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# ss = np.hstack((img, gradient))
# cv2.imwrite('seg_gr.png', gradient)

# img=cv2.imread('seg_gr.png',0)
# print(np.unique(img))
# -----------------------------------------将一张图片裁剪成多张小图片------------------------------
# from PIL import Image
#
# img = Image.open("./lena.png")
# print(img.size)
# h,w=img.size
# x0 = 0
# y0 = 0  # 起点坐标，作为变量方便调整起始位置
# dx = h/4;
# dy = w/4;  # 裁剪范围
# for col in range(4):  # 列
#     for row in range(4):  # 行
#         cropped = img.crop(
#             (x0 + dx * col, y0 + dy * row, x0 + dx * (col + 1), y0 + dy * (row + 1)))  # (left, upper, right, lower)
#         cropped.save("./crop/Sample_{}{}.tif".format(row, col))

# ---------------------------------------------------------------------
'''
读入一个图片0.bmp，切成指定数目个小图片(16个)
文件夹名out
'''
from PIL import Image
import sys, os

cut_num = 4  # 4*4=16个图片


# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    # new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    new_image = Image.new(image.mode, (new_image_length, new_image_length))
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
    # (x,y)二元组表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


# 切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / cut_num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, cut_num):  # 两重循环，生成图片基于原图的位置
        for j in range(0, cut_num):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list


# 保存
def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('out/' + str(index) + '.bmp', 'BMP')
        index += 1


# if __name__ == '__main__':
#     file_path = "lena.png"
#     os.makedirs("out",exist_ok=True)
#     image = Image.open(file_path)
#     # image.show()
#     image = fill_image(image)
#     image_list = cut_image(image)
#     save_images(image_list)

# -------------------------------------二、随机截取指定大小的图
#
# '''
# 随即截取指定大小的图片
# '''
# import os
# import cv2
# import random
#
# #读取图片
# img1=cv2.imread('01_2.png')
# img2=cv2.imread('01_2label.png')
#
# #h、w为想要截取的图片大小
# h=224
# w=224
#
# save_dir1 = "crop/img/"
# save_dir2 = "crop/mask/"
# if os.path.exists(save_dir1) is False:
#      os.makedirs(save_dir1)
# if os.path.exists(save_dir2) is False:
#      os.makedirs(save_dir2)
# count=0
# while 1:
#      #随机产生x,y 此为像素内范围产生
#      y = random.randint(0, 224)
#      x = random.randint(0, 224)
#      #随机截图
#      cropImg1 = img1[(y):(y + h), (x):(x + w)]
#      cropImg2 = img2[(y):(y + h), (x):(x + w)]
#      cv2.imwrite(save_dir1 + str(count) + '.bmp', cropImg1)
#      cv2.imwrite(save_dir2 + str(count) + '.bmp', cropImg2)
#      count+=1
#
#      if count==100:
#         break

# ----------------------------------------------------------------三、小图组合成大图

# '''
# 将指定文件夹里面的图片拼接成一个大图片
# '''
# import PIL.Image as Image
# import os
#
# IMAGES_PATH = 'crop\\img\\'  # 图片集地址
# IMAGES_FORMAT = ['.bmp', '.BMP']  # 图片格式
# IMAGE_SIZE = 224  # 每张小图片的大小
# IMAGE_ROW = 10  # 图片间隔，也就是合并成一张图后，一共有几行
# IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列
# IMAGE_SAVE_PATH = 'final.bmp'  # 图片转换后的地址
#
# # 获取图片集地址下的所有图片名称
# image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#                os.path.splitext(name)[1] == item]
#
# # 简单的对于参数的设定和实际图片集的大小进行数量判断
# if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
#     raise ValueError("合成图片的参数和要求的数量不能匹配！")
#
#
# # 定义图像拼接函数
# def image_compose():
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
#             to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
#     # to_image = to_image.convert('L')#以灰度图存储
#
#     return to_image.save(IMAGE_SAVE_PATH)  # 保存新图
#
#
# image_compose()  # 调用函数