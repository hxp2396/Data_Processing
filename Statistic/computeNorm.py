# -*- coding: utf-8 -*-计算灰度数据集的均值和方差
# @Time    : 2021/2/24 17:34
# @Author  : tingsong
# @FileName: compute_mean_std.py
from PIL import Image
import cv2, os
import numpy as np
from tqdm import tqdm
def computeNorm(input_path):
    means, stdevs = [], []
    for patient in os.listdir(input_path):
        patient_path = os.path.join(input_path, patient)
        img_path = patient_path
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        img = img.astype(np.float32) / 255.
        means.append(img[:, :].mean())
        stdevs.append(img[:, :].std())
    means = np.mean(means)
    stdevs = np.mean(stdevs)
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
def computeNorm_2(dir):
    img_filenames = os.listdir(dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(dir + '/' + img_filename)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])
def computeNorm_3(img_h, img_w,imgs_path):
    means, stdevs = [], []
    img_list = []
    imgs_path_list = os.listdir(imgs_path)
    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

if __name__ == '__main__':
    input_path= r'E:\DataCollection\Covid\train\Images'
    computeNorm(input_path)