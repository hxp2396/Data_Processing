# !/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author: hxp2396
# @Time : 2024/10/14 10:14
# @File : splitdataset.py
# @annotation:划分数据集
import os
import random
import shutil
# 划分数据集
def organize_dataset(dataset_path, images_path, labels_path, train_ratio, valid_ratio):
    images_name = os.listdir(images_path)
    images_num = len(images_name)
    alpha = int(images_num * train_ratio)
    beta = int(images_num * (train_ratio + valid_ratio))
    print(images_num)
    random.shuffle(images_name)
    train_list = images_name[0:alpha]
    valid_list = images_name[alpha:beta]
    test_list = images_name[beta:images_num]
    # 确认分割正确
    print('train list: ', len(train_list))
    print('valid list: ', len(valid_list))
    print('test list: ', len(test_list))
    print('total num: ', len(test_list) + len(valid_list) + len(train_list))
    # 创建train,valid和test的文件夹
    train_images_path = os.path.join(dataset_path, 'train_images')
    train_labels_path = os.path.join(dataset_path, 'train_labels')
    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(train_labels_path) == False:
        os.mkdir(train_labels_path)
    valid_images_path = os.path.join(dataset_path, 'valid_images')
    valid_labels_path = os.path.join(dataset_path, 'valid_labels')
    if os.path.exists(valid_images_path) == False:
        os.mkdir(valid_images_path)
    if os.path.exists(valid_labels_path) == False:
        os.mkdir(valid_labels_path)
    test_images_path = os.path.join(dataset_path, 'test_images')
    test_labels_path = os.path.join(dataset_path, 'test_labels')
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)
    if os.path.exists(test_labels_path) == False:
        os.mkdir(test_labels_path)
    # 拷贝影像到指定目录
    for image in train_list:
        shutil.copy(os.path.join(images_path, image), os.path.join(train_images_path, image))
        shutil.copy(os.path.join(labels_path, image), os.path.join(train_labels_path, image))
    for image in valid_list:
        shutil.copy(os.path.join(images_path, image), os.path.join(valid_images_path, image))
        shutil.copy(os.path.join(labels_path, image), os.path.join(valid_labels_path, image))
    for image in test_list:
        shutil.copy(os.path.join(images_path, image), os.path.join(test_images_path, image))
        shutil.copy(os.path.join(labels_path, image), os.path.join(test_labels_path, image))


if __name__ == '__main__':
    # 数据集路径
    dataset_path = r'E:\DataCollection_new\IDS_Subcutaneous'
    images_path = r'E:\DataCollection_new\IDS_Subcutaneous\Image'
    labels_path = r'E:\DataCollection_new\IDS_Subcutaneous\Label'
    train_ratio = 0.8
    valid_ratio = 0.1
    organize_dataset(dataset_path, images_path, labels_path, train_ratio, valid_ratio)
